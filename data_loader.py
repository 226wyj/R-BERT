import os
import csv
import copy
import json
import logging

# pytorch
import torch
from torch.utils.data import TensorDataset

from utils import get_label

logger = logging.getLogger(__name__)


class InputExample(object):
    """若想调用__dict__和__repr__等方法，必须继承object对象
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    # 自定义输出实例化对象的格式，这里输出json字符串的格式
    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    """

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id,
                 e1_mask, e2_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        indent: 缩进
        sort_keys: 键值按照字典序排序
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Processor(object):
    """ Base Processor """
    
    def __init__(self, args):
        self.args = args
        self.relation_labels = get_label(args)

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file.
        Return: 二维列表
            [
                [relation1, text1],
                [relation2, text2],
                ...
            ]
        """
        with open(input_file, "r", encoding="utf-8") as f:
            # delimiter: 用于分隔字段的单字符串，默认为','
            # quotechar: 用于带有特殊字符（如分隔符）的字段引用符号，默认为 "
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(self, lines, set_type):
        """
        Creates examples for the training and dev sets.
        set_type: train / dev / test
        """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)  # 组合成guid，如：train-1, test-4等
            try:
                text_a = line[1]                # line[0]是关系类别，line[1]是完整句子
            except IndexError:
                raise Exception("Index Error!\nproblem data's id: {}\nproblem data's content: {}".format(i, line))
            if not self.args.no_lower_case:
                text_a = text_a.lower()
            label = self.relation_labels.index(line[0]) # 对应关系在标签列表中的下标，类似于id
            if i % 1000 == 0:               # 每处理1000行在日志中输出一条信息
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))    # json格式字符串
        return examples


class SemEvalProcessor(Processor):
    """
    Processor for the Semeval Dataset 
    """

    def __init__(self, args):
        Processor.__init__(self, args)

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
            
        logger.info("LOOKING AT {}".format(os.path.join(self.args.semeval_data_dir, file_to_read)))

        return self._create_examples(self._read_tsv(os.path.join(self.args.semeval_data_dir, file_to_read)), mode)
    
class FinancialProcessor(Processor):
    """
    Processor for Financial Dataset
    """
    def __init__(self, args):
        Processor.__init__(self, args)
    
    def get_examples(self, mode):
        """获取example"""
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        logger.info("LOOKING AT {}".format(os.path.join(self.args.financial_data_dir, file_to_read)))
        return self._create_examples(
            lines=self._read_tsv(os.path.join(self.args.financial_data_dir, file_to_read)),
            set_type=mode
        )

class PeopleProcessor(Processor):
    """
    Processor for People Dataset
    """
    def __init__(self, args):
        Processor.__init__(self, args)
    
    def get_examples(self, mode):
        """获取example"""
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file
        logger.info("LOOKING AT {}".format(os.path.join(self.args.people_data_dir, file_to_read)))
        return self._create_examples(
            lines=self._read_tsv(os.path.join(self.args.people_data_dir, file_to_read)),
            set_type=mode
        )

processors = {
    "semeval": SemEvalProcessor,
    "financial": FinancialProcessor,
    "people": PeopleProcessor
}

# 处理中文
def convert_chinese_examples(examples, max_seq_len, tokenizer,
                             cls_token='[CLS]',
                             cls_token_segment_id=0,
                             sep_token='[SEP]',
                             pad_token=0,
                             pad_token_segment_id=0,
                             sequence_a_segment_id=0,
                             add_sep_token=False,
                             mask_padding_with_zero=True):
    features = []
    datas = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        
        data = {}
        data['句子'] = example.text_a
        data['关系'] = int(example.label)

        # 分词
        tokens_a = tokenizer.tokenize(example.text_a)
        
        # 标识实体分隔符的位置
        e11_p = tokens_a.index("<e1>")
        e12_p = tokens_a.index("</e1>")
        e21_p = tokens_a.index("<e2>")
        e22_p = tokens_a.index("</e2>")
        
        # 替换分隔符
        tokens_a[e11_p] = '$'
        tokens_a[e12_p] = '$'
        tokens_a[e21_p] = '#'
        tokens_a[e22_p] = '#'

        # 根据分隔符的位置推算出实体的位置, 需要注意的是, 
        # 由于要在开头插入[CLS]，因此所有实体的前分隔符下标还要顺次+1
        e11_p += 2
        e21_p += 2

        # 特殊标识符计数，用于后续填充句子用
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        # 长度超标，则判断实体位置，若实体会被截断则舍弃该数据
        if len(tokens_a) > max_seq_len - special_tokens_count:
            if e12_p >= max_seq_len or e22_p >= max_seq_len:
                continue
            else:
                tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

        tokens = tokens_a
        if add_sep_token:
            tokens += [sep_token]
        
        # token_type_id是在做问答系统时用于标识Q和A的，这里我们不需要，因此统一取0
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # token列表开头加入[CLS]，同时更新token_type_ids
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        # 根据vocabulary将对应的汉字转换为唯一的id编号
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 由于所有的句子需要经过填充变为统一长度，因此喂给模型的句子中
        # 并非所有的信息都是有用信息，还有可能是填充的无用字符。
        # attention_mask为1个与输入等长的列表，默认用1标识有用信息，0标识无用信息
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # 需要填充的长度
        padding_length = max_seq_len - len(input_ids)
        # 填充句子id和对应的attention_mask
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        
        # e1 mask, e2 mask, 用于标识两个实体的位置
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1    

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = int(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))
        

        datas.append(data)
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_id=label_id,
                e1_mask=e1_mask,
                e2_mask=e2_mask
            )
        )
    print(len(features), len(datas))
    assert len(features) == len(datas)
    return datas, features

# 处理英文
def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=0,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 add_sep_token=False,
                                 mask_padding_with_zero=True,
                                 replace_seperate_token=False,
                                 del_seperate_token=False):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))


        # 分词
        tokens_a = tokenizer.tokenize(example.text_a)

        # 分隔符标注
        e11_p = tokens_a.index("<e1>")  # the start position of entity1
        e12_p = tokens_a.index("</e1>")  # the end position of entity1
        e21_p = tokens_a.index("<e2>")  # the start position of entity2
        e22_p = tokens_a.index("</e2>")  # the end position of entity2

        # Replace the token
        tokens_a[e11_p] = "$"
        tokens_a[e12_p] = "$"
        tokens_a[e21_p] = "#"
        tokens_a[e22_p] = "#"

        if replace_seperate_token:  
            # 实验：采用其他符号替代分隔符是否会影响结果
            
            tokens_a[e11_p] = "@"
            tokens_a[e12_p] = "@"
            tokens_a[e21_p] = "&"
            tokens_a[e22_p] = "&"

        # Add 1 because of the [CLS] token
        e11_p += 2
        e21_p += 2

        if del_seperate_token:        
            # 实验：删除特殊符号对结果的影响
            
            if e11_p < e21_p:           # 实体1在实体2前面
                
                del tokens_a[e11_p]     # 删除<e1>, 则e11_p指向实体1的第一个字符，不需要进一步调整下标
                
                del tokens_a[e12_p - 1] # 删除</e1>, 此时由于已经删除了<e1>，则e12_p指向实体1之后的第一个字符，需要调整下标
                e12_p -= 2              # 由于删除了<e1>和</e1>，e12_p属于向后移了2位，需要调整下标
                
                del tokens_a[e21_p - 2] # 删除<e2>
                e21_p -= 2

                del tokens_a[e22_p - 3] # 删除</e2>
                e22_p -= 4
            
            elif e21_p < e11_p:         # 实体2在实体1前面
                
                del tokens_a[e21_p]     # 删除<e2>, 则e21_p指向实体1的第一个字符，不需要进一步调整下标
                
                del tokens_a[e22_p - 1] # 删除</e2>, 此时由于已经删除了<e2>，则e22_p指向实体1之后的第一个字符，需要调整下标
                e22_p -= 2              # 由于删除了<e2>和</e2>，e22_p属于向后移了2位，需要调整下标
                
                del tokens_a[e11_p - 2] # 删除<e1>
                e11_p -= 2

                del tokens_a[e12_p - 3] # 删除</e1>
                e12_p -= 4


        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for R-BERT.
        if add_sep_token:
            special_tokens_count = 2
        else:
            special_tokens_count = 1
        
        # token过长则截断，反之说明需要填充
        if len(tokens_a) > max_seq_len - special_tokens_count:
            tokens_a = tokens_a[:(max_seq_len - special_tokens_count)]

        tokens = tokens_a
        if add_sep_token:
            tokens += [sep_token]

        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # 句首插入[CLS]
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # e1 mask, e2 mask
        e1_mask = [0] * len(attention_mask)
        e2_mask = [0] * len(attention_mask)

        for i in range(e11_p, e12_p + 1):
            e1_mask[i] = 1
        for i in range(e21_p, e22_p + 1):
            e2_mask[i] = 1

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)

        label_id = int(example.label)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))
            logger.info("e1_mask: %s" % " ".join([str(x) for x in e1_mask]))
            logger.info("e2_mask: %s" % " ".join([str(x) for x in e2_mask]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id,
                          e1_mask=e1_mask,
                          e2_mask=e2_mask))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    """ 将文本文档中的数据加载出来并整理成DataSet形式 """
    processor = processors[args.task](args)

    # Load data features from cache or dataset file
    if args.task == 'financial':
        data_path = args.financial_data_dir
    elif args.task == 'semeval':
        data_path = args.semeval_data_dir   
    elif args.task == 'people':
        data_path = args.people_data_dir
    cached_features_file = os.path.join(data_path, 'cached_{}_{}'.format(args.task, mode))

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_path)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")
        
        if args.task == 'semeval':
            features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token)
        elif args.task == 'financial' or args.task == 'people':
            datas, features = convert_chinese_examples(examples, args.max_seq_len, tokenizer, add_sep_token=args.add_sep_token)
            # 更新文件
            labels = get_label(args)
            with open(os.path.join(data_path, mode + ".tsv"), 'w', encoding='utf-8 ') as f:
                for data in datas:
                    f.write("{}\t{}\n".format(labels[data['关系']], data['句子'].replace('\n', '')))

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in features], dtype=torch.long)  # add e2 mask

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_label_ids, all_e1_mask, all_e2_mask)
    return dataset
