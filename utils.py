import sys
sys.path.append('.')

import os
import random
import logging
import collections
import csv
from copy import deepcopy

import torch
import numpy as np
from sklearn.metrics import f1_score
from transformers.tokenization_bert import BertTokenizer

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sn 
import pandas as pd 

def get_special_tokens():
    input_file = './DataSets/Financial/special_tokens.txt'
    tokens = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tokens.append(line.replace('\n', ''))
    return tokens

ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]

# 检查各个路径是否存在，若不存在则建立
def check_paths(args):
    # Semeval
    if not os.path.exists(args.semeval_data_dir):
        os.makedirs(args.semeval_data_dir)
    if not os.path.exists(args.semeval_eval_dir):
        os.makedirs(args.semeval_eval_dir)
    if not os.path.exists(args.semeval_model_dir):
        os.makedirs(args.semeval_model_dir)
    # Financial
    if not os.path.exists(args.financial_data_dir):
        os.makedirs(args.financial_data_dir)
    if not os.path.exists(args.financial_eval_dir):
        os.makedirs(args.financial_eval_dir)
    if not os.path.exists(args.financial_model_dir):
        os.makedirs(args.financial_model_dir)
    # People
    if not os.path.exists(args.people_data_dir):
        os.makedirs(args.people_data_dir)
    if not os.path.exists(args.people_eval_dir):
        os.makedirs(args.people_eval_dir)
    if not os.path.exists(args.people_model_dir):
        os.makedirs(args.people_model_dir)

# 获取全部的关系，返回一个列表
def get_label(args):
    if args.task == 'semeval':
        path = args.semeval_data_dir
    elif args.task == 'financial':
        path = args.financial_data_dir
    elif args.task == 'people':
        path = args.people_data_dir
    return [label.strip() for label in open(os.path.join(path, args.label_file), 'r', encoding='utf-8')]

# 英文数据集的tokenizer
def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    return tokenizer

# 中文数据集的tokenizer
def load_chinese_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.chinese_vocab)
    tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
    # print(ADDITIONAL_SPECIAL_TOKENS)
    return tokenizer

# 用于SemEval的官方测试脚本，需要将测试结果按一定格式写入文件
def write_prediction(args, output_file, preds, write_id=False, id=8001):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, 'w', encoding='utf-8') as f:
        if write_id:
            for idx, pred in enumerate(preds):
                f.write("{}\t{}\n".format(id + idx, relation_labels[pred]))
        else:
            for pred in preds:
                f.write("{}\n".format(relation_labels[pred]))

# 保存中文测试数据结果
def write_chinese_prediction(args, output_file, preds):
    relation_labels = get_label(args)
    with open(output_file, 'w', encoding="utf-8") as f:
        for pred in preds:
            f.write("{}\n".format(relation_labels[pred]))

def init_logger():
    """将日志信息输出到控制台
    Params:
        asctime: 打印日志的时间
        levelname: 打印日志级别
        name: 打印日志名字
        message: 打印日志信息 
    """
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    """
    为了得到可重复的实验结果需要对所有随机数生成器设置一个固定的种子
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    """ 返回准确率P以及F1值，这里取的是macro F1 
    Params:
        preds:  预测结果
        labels: 标签
    """
    assert len(preds) == len(labels)
    acc = (preds == labels).mean()
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
    }

def calculate_f1(args, confusion_matrix_file='confusion_matrix.txt', evaluation_result_file='evaluation_result.txt'):
    """ 手动计算各类关系的P, R, micro-F1, macro-F1 """

    # 获取测试集和对应的预测结果
    if args.task == 'semeval':
        test_dir = args.semeval_data_dir
        predict_dir = args.semeval_eval_dir
    elif args.task == 'financial':
        test_dir = args.financial_data_dir
        predict_dir = args.financial_eval_dir
    elif args.task == 'people':
        test_dir = args.people_data_dir
        predict_dir = args.people_eval_dir

    test_file = 'test.tsv'
    if args.task == 'semeval':
        predict_file = 'predicted_answers_for_f1.txt'
    else:
        predict_file = 'predicted_answers.txt'

    # 文件路径
    test_path = os.path.join(test_dir, test_file)
    predict_path = os.path.join(predict_dir, predict_file)
    

    # 获取测试集中的数据
    with open(test_path, 'r', encoding='utf-8') as f:
        datas = [line.replace('\n', '') for line in f.readlines()]

    # 获取全部的标签
    labels = get_label(args)

    # 混淆矩阵, 设有n类关系, 则矩阵为n * n维
    num_labels = len(labels)
    confusion_matrix = np.zeros([num_labels, num_labels])

    # 按顺序获取全部预测结果, 要注意把每行末尾的换行符去掉
    results = []
    with open(predict_path, 'r', encoding='utf-8') as f:
        results = [line.replace('\n', '') for line in f.readlines()]
        # for line in f.readlines():
        #     data = line.replace('\n', '')
        #     results.append(data)

    assert len(datas) == len(results)

    # 根据预测结果更新混淆矩阵
    for i in range(len(results)):
        predict_result = results[i]
        predict_index = labels.index(predict_result)
        real_result = datas[i].split('\t')[0]
        real_index = labels.index(real_result)
        confusion_matrix[predict_index][real_index] += 1

    # 存储混淆矩阵, 和预测结果存在相同目录下
    np.savetxt(os.path.join(predict_dir, confusion_matrix_file), confusion_matrix, fmt="%d")

    # 针对每类关系分别计算P, R, F1
    # 混淆矩阵中每一行代表真实标签，每一列代表预测结果
    # 所以在行方向上相加(axis=0)得到的是预测的聚类，列方向上相加(axis=1)得到的是标签的聚类
    evaluation = collections.OrderedDict()
    precision_sum = confusion_matrix.sum(axis=0)
    recall_sum = confusion_matrix.sum(axis=1)

    # 结果以字典形式返回
    # {
    #   relation1: {P: , R: , F1: },
    #   relation2: {P: , R: , F1: },
    #   ...
    # }
    for i in range(len(labels)):
        relation = labels[i]
        evaluation[relation] = {}
        precision = confusion_matrix[i][i] / precision_sum[i]
        recall = confusion_matrix[i][i] / recall_sum[i]
        f1 = 2 * precision * recall / (precision + recall)

        evaluation[relation]['P'] = precision
        evaluation[relation]['R'] = recall
        evaluation[relation]['F1'] = f1  

    for k, v in evaluation.items():
        print("{} -> P:{} R:{} F1:{}".format(k, v['P'], v['R'], v['F1']))
    
    # macro-F1 (excluding 'NULL')
    # 在原混淆矩阵的基础上分别求出每类关系的P和R(除了NULL关系)
    # 然后求出所有P和R的平均值，利用平均值求F1值
    macro_p = 0.0
    macro_r = 0.0
    for k, v in evaluation.items():
        if k == 'NULL':
            continue
        else:
            macro_p += v['P']
            macro_r += v['R']
    macro_p /= (len(labels) -1) 
    macro_r /= (len(labels) -1) 
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r)

    # micro-F1 (excluding 'NULL')
    # 统计各类标(除了NULL关系)的TP, FP, TN, FN, 分别加和构成新的混淆矩阵
    # 根据新的混淆矩阵求F1值 
    micro_TP = 0
    for i in range(len(labels)):
        if i == labels.index('NULL'):
            continue
        else:
            micro_TP += confusion_matrix[i][i]
    micro_p_sum = precision_sum.sum() - precision_sum[labels.index('NULL')]
    micro_r_sum = recall_sum.sum() - recall_sum[labels.index('NULL')]
    mirco_p = micro_TP / micro_p_sum
    micro_r = micro_TP / micro_r_sum
    micro_f1 = 2 * mirco_p * micro_r / (mirco_p + micro_r)

    print("macro-p:{} \t macro-r:{} \t macro-f1:{}".format(macro_p, macro_r, macro_f1))
    print("micro-p:{} \t micro-r:{} \t micro-f1:{}".format(mirco_p, micro_r, micro_f1))

    # 将评测结果写入文件保存
    with open(os.path.join(predict_dir, evaluation_result_file), 'w', encoding='utf-8') as f:
        for k, v in evaluation.items():
            f.write("{}\t\t\tP: {}\tR: {}\tF1: {}\n".format(k, v['P'], v['R'], v['F1']))
        f.write("\nmacro-p: {} \t macro-r: {} \t macro-f1: {}".format(macro_p, macro_r, macro_f1))
        f.write("\nmicro-p: {} \t micro-r: {} \t micro-f1: {}".format(mirco_p, micro_r, micro_f1))

    return confusion_matrix

def draw_heatmap(test_file, matrix_file):
    """ 读取混淆矩阵文件并将其绘制为热力图 """

    # 读取测试集
    counts = collections.OrderedDict()
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            relation = line.split('\t')[0]
            if relation in counts:
                counts[relation] += 1
            else:
                counts[relation] = 1
    
    # 获取标签
    labels = counts.keys()

    # 读取混淆矩阵
    confusion_matrix = np.loadtxt(matrix_file, dtype=int)

    plt.rcParams['font.sans-serif'] = ['SimHei']    # 中文字体设置-黑体
    plt.rcParams['axes.unicode_minus'] = False      # 解决保存图像是负号'-'显示为方块的问题
    sn.set(font='SimHei')                           # 解决Seaborn中文显示问题

    df_cm = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    # plt.figure(figsize=(15, 15))
    sn.heatmap(df_cm, fmt='d', annot=True, linewidth=.5, cmap="YlGnBu") 

    plt.show()
