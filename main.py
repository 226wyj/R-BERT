import argparse
import os 

from trainer import Trainer
from utils import check_paths, init_logger, load_tokenizer, load_chinese_tokenizer
from data_loader import load_and_cache_examples


def main(args):

    init_logger()
    check_paths(args)

    if args.task == "semeval":
        tokenizer = load_tokenizer(args)            # semeval
    elif args.task == "financial" or args.task == "people":
        tokenizer = load_chinese_tokenizer(args)    # 金融数据


    train_dataset = None
    dev_dataset = None
    test_dataset = None

    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    if args.do_dev:
        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    if args.do_eval:
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset)

    if args.do_train:
        trainer.train()

    # if args.do_dev:
    #     trainer.load_model()
    #     trainer.evaluate('dev')

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate('test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 任务类型: [semeval, financial, people]
    parser.add_argument("--task", default="financial", type=str, help="The name of the task to train: [semeval, financial, people]")

    # Semeval数据的文件路径
    parser.add_argument("--semeval_data_dir", default="./DataSets/SemEval", type=str,
                        help="The input dir of SemEval. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--semeval_eval_dir", default="./DataSets/SemEval/eval", type=str, help="Evaluation script, result directory")
    parser.add_argument("--semeval_model_dir", default="./model/semeval", type=str, help="Path to Semeval model")

    # 中文金融数据的文件路径
    parser.add_argument("--financial_data_dir", default="./DataSets/Financial", type=str, help="Chinese data dir, contain .tsv files")
    parser.add_argument("--financial_eval_dir", default="./DataSets/Financial/eval", type=str, help="Prediction output file path used for evaluating")
    parser.add_argument("--financial_model_dir", default="./model/financial", type=str, help="Path to financial model")

    # 人物关系数据的文件路径
    parser.add_argument("--people_data_dir", default="./DataSets/People", type=str, help="people relation data dir, contain .tsv files")
    parser.add_argument("--people_eval_dir", default="./DataSets/People/eval", type=str, help="Prediction output file path used for evaluating")
    parser.add_argument("--people_model_dir", default="./model/people", type=str, help="Path to people model")

    # 训练集、验证集、测试集、标签
    parser.add_argument("--train_file", default="train.tsv", type=str, help="Train file")
    parser.add_argument("--dev_file", default="dev.tsv", type=str, help="Validation file")
    parser.add_argument("--test_file", default="test.tsv", type=str, help="Test file")
    parser.add_argument("--label_file", default="label.txt", type=str, help="Label file")

    # bert-base-uncased模型，用于英文数据集分类
    parser.add_argument("--pretrained_model_name", default="bert-base-uncased", required=False, help="Pretrained model name")
    # bert-base-chinese模型，用于中文数据集，本地模型
    parser.add_argument("--pretrained_chinese_model", default="./bert-base-chinese", required=False, help="Chinese pretrained model path")
    # chinese-vocabulary，中文词汇表，用于tokenizer分词
    parser.add_argument("--chinese_vocab", default="./bert-base-chinese/vocab.txt", required=False, help="Chinese vocab list")

    # 超参数
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation.")
    parser.add_argument("--max_seq_len", default=200, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    # 输出日志 / 保存模型的步长
    parser.add_argument('--logging_steps', type=int, default=250, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=250, help="Save checkpoint every X updates steps.")

    # bool值默认为false，当命令中包含如下参数时则bool值变为true
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_dev", action="store_true", help="Whether to validate while training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_lower_case", action="store_true", help="Whether not to lowercase the text (For cased model)")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--add_sep_token", action="store_true", help="Add [SEP] token at the end of the sentence")

    args = parser.parse_args()
    main(args)
