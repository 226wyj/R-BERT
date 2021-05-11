import os
import logging
from tqdm import tqdm, trange
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from model import RBERT
from utils import set_seed, write_prediction, write_chinese_prediction, compute_metrics, get_label

logger = logging.getLogger(__name__)


class Trainer(object):
    """训练器
    Params:
        args: 命令行参数
        train_dataset: 训练集
        dev_dataset: 验证集
        test_dataset: 测试集
        label_lst: 所有的标签类别
        num_labels: 标签总数
        bert_config: BERT模型的初始化器
        model: 神经网络模型，这里取R-BERT
        device: cpu 或者是 cuda
    """
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.label_lst = get_label(args)
        self.num_labels = len(self.label_lst)

        if self.args.task == "semeval":
            self.bert_config = BertConfig.from_pretrained(args.pretrained_model_name, num_labels=self.num_labels, finetuning_task=args.task)
        else:
            self.bert_config = BertConfig.from_pretrained(args.pretrained_chinese_model, num_labels=self.num_labels, finetuning_task=args.task)

        self.model = RBERT(self.bert_config, args)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        """ 训练模型 """

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        f1 = 0

        # t_total: 累计更新参数的次数
        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and scheduler (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        set_seed(self.args) # 固定随机数种子

        for i in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", ncols=70)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                        #   'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5]}
                labels = batch[3]
                outputs = self.model(**inputs)
                logits = outputs[0]

                # 交叉熵计算误差
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))

                # 若梯度累加一定次数后再更新，则loss需要除以累加的次数以获得平均值
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()

                # 达到梯度累加的次数，进行参数更新
                if (step + 1) % self.args.gradient_accumulation_steps == 0:

                    # 梯度裁剪，防止BP过程的梯度爆炸 / 消失
                    # 将gradient的值限制在[-max_grad_norm, max_grad_norm]之间
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    # 更新参数
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                    
                    # 在验证集上进行验证，由于semeval数据集没有验证集，因此在test集上做验证
                    # if self.args.do_dev and self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                    #     if self.args.task == "semeval":
                    #         self.evaluate('test')
                    #     else:
                    #         self.evaluate('test')

                    # 保存模型策略： 训练的前2轮直接保存，Epoch > 2之后先在验证集上进行验证，若f1值高于之前的模型就保存，反之则丢弃
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        if i > 1 and self.args.do_dev:
                            # self.load_model()
                            macro_f1 = self.evaluate(mode='dev')['f1']
                            if macro_f1 > f1:
                                f1 = macro_f1
                                self.save_model()
                        else:
                            self.save_model()

        return global_step, tr_loss / global_step

    def evaluate(self, mode):
        """测试模型准确率 
        Params:
            mode: test / dev -> 测试 / 验证
        """

        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        num_eval_steps = 0
        preds = np.zeros(len(self.label_lst))
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=70):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2],
                        #   'labels': batch[3],
                          'e1_mask': batch[4],
                          'e2_mask': batch[5]}
                labels = batch[3]
                outputs = self.model(**inputs)

                # tmp_eval_loss, logits = outputs[:2]

                logits = outputs[0]
                criterion = nn.CrossEntropyLoss()
                tmp_eval_loss = criterion(logits.view(-1, self.num_labels), labels.view(-1))


                eval_loss += tmp_eval_loss.mean().item()
            num_eval_steps += 1

            if num_eval_steps == 1:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            elif preds is None:
                print(num_eval_steps)
                return
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0)
        
        # np.savetxt('./DataSets/Financial/preds.txt', preds)

        eval_loss = eval_loss / num_eval_steps
        results = {
            "loss": eval_loss
        }
        preds = np.argmax(preds, axis=1)    # 和softmax一样，求得最大的概率，只是写法不同

        # np.savetxt('./DataSets/Financial/preds_softmax.txt', preds)


        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("%s = %s", key, str(results[key]))

        # 测试时才保存结果，验证时无需保存
        if mode == "test":
            if self.args.task == "semeval":
                write_prediction(self.args, os.path.join(self.args.semeval_eval_dir, "predicted_answers.txt"), preds, write_id=True, id=8001)
                write_prediction(self.args, os.path.join(self.args.semeval_eval_dir, "predicted_answers_for_f1.txt"), preds)
            elif self.args.task == "financial":
                write_prediction(self.args, os.path.join(self.args.financial_eval_dir, "predicted_answers.txt"), preds)
                # write_chinese_prediction(self.args, os.path.join(self.args.financial_eval_dir, "predicted_answers.txt"), preds)
            elif self.args.task == "people":
                write_prediction(self.args, os.path.join(self.args.people_eval_dir, "predicted_answers.txt"), preds)
                # write_chinese_prediction(self.args, os.path.join(self.args.people_eval_dir, "predicted_answers.txt"), preds)
        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)

        if self.args.task == "semeval":
            output_dir = os.path.join(self.args.semeval_model_dir)
        elif self.args.task == "financial":
            output_dir = os.path.join(self.args.financial_model_dir)
        elif self.args.task == "people":
            output_dir = os.path.join(self.args.people_model_dir)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)

    def load_model(self):
        # Check whether model exists
        if self.args.task == 'semeval' and not os.path.exists(self.args.semeval_model_dir):
            raise Exception("Model doesn't exists! Train first!")
        elif self.args.task == "financial" and not os.path.exists(self.args.financial_model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            if self.args.task == 'semeval':
                model_dir = self.args.semeval_model_dir
            elif self.args.task == 'financial':
                model_dir = self.args.financial_model_dir
            elif self.args.task == 'people':
                model_dir = self.args.people_model_dir
                
            self.bert_config = BertConfig.from_pretrained(model_dir)
            logger.info("***** Bert config loaded *****")
            self.model = RBERT.from_pretrained(model_dir, config=self.bert_config, args=self.args)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            traceback.print_exc()
            # raise Exception("Some model files might be missing...")
