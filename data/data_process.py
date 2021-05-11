import sys
sys.path.append('.')

from semeval_processor import SemEvalProcessor
import platform
import os 
import argparse

def ProcessSemData():
    """
    处理SemEval数据集 
    """
    dataProcessor = SemEvalProcessor()

    # 处理训练数据集
    TRAIN_FILE = './SemEval/origin/TRAIN_FILE.TXT'
    RELATION_ID_FILE = './SemEval/processed/relation_id.txt'
    TRAIN_SENTENCE_FILE = './SemEval/processed/train_sentences.txt'
    TRAIN_COORESPONDING_RELATION_FILE = './SemEval/processed/train_cooresponding_relation.txt'
    TRAIN_RELATION_SENTENCE_FILE = './SemEval/train.tsv'

    dataProcessor.relation_2_id(TRAIN_FILE, RELATION_ID_FILE)
    dataProcessor.format_process(TRAIN_FILE, TRAIN_SENTENCE_FILE, TRAIN_COORESPONDING_RELATION_FILE)
    dataProcessor.combine_sentence_relation(TRAIN_SENTENCE_FILE, TRAIN_COORESPONDING_RELATION_FILE, TRAIN_RELATION_SENTENCE_FILE)

    # 处理测试数据集
    TEST_FILE = './SemEval/origin/TEST_FILE_FULL.TXT'
    TEST_SENTENCE_FILE = './SemEval/processed/test_sentences.txt'
    TEST_COORESPONDING_RELATION_FILE = './SemEval/processed/test_cooresponding_relation.txt'
    TEST_RELATION_SENTENCE_FILE = './SemEval/test.tsv'

    dataProcessor.format_process(TEST_FILE, TEST_SENTENCE_FILE, TEST_COORESPONDING_RELATION_FILE)
    dataProcessor.combine_sentence_relation(TEST_SENTENCE_FILE, TEST_COORESPONDING_RELATION_FILE, TEST_RELATION_SENTENCE_FILE)

    # 获取所有关系类别
    LABEL = './SemEval/label.txt'
    dataProcessor.getLabels(RELATION_ID_FILE, LABEL)

def ProcessPeopleData():
    """ 
    处理人物关系数据集 
    """
    people_data_path = './People/origin/all_data.txt'

    processor = PeopleProcessor(people_data_path)

    processor.relation_counts()

    processor.data_cleaning()

    processor.data_formatting()

    processor.divide_datasets(do_dev=True, balance_sample=False)


def transferProcessedData(task=1):
    """将处理好的数据文件转移到指定目录下
    Params:
        task=0: SemEval数据集
        task=1: 金融实体关系抽取
        task=2: 人物关系数据集
    """
    if task not in [0, 1, 2]:
        raise Exception("Error! The task can only be 1: Semeval, 2: Financial and 3: People!")

    platform_name = platform.system()
    # 各个文件名字
    if task == 0:
        files = ["label.txt", "train.tsv", "test.tsv"]
    elif task == 1 or task == 2:
        files = ["label.txt", "train.tsv", "test.tsv", "dev.tsv"]

    # Financial源文件目录
    F_Root = 'Financial'
    # Financial目标文件目录
    F_Destiny_Win = '..\\DataSets\\Financial'
    F_Destiny_Lin = '../DataSets/Financial'

    # SemEval源文件目录
    S_Root = 'SemEval'
    # SemEval目标文件目录
    S_Destiny_Win = '..\\DataSets\\SemEval'
    S_Destiny_Lin = '../DataSets/SemEval'

    # People源文件目录
    P_Root = 'People'
    # People目标文件目录
    P_Destiny_Win = '..\\DataSets\\People'
    P_Destiny_Lin = '../DataSets/People'

    # 将源目录中的文件复制到目标目录下
    for file_name in files:
        if platform_name == "Windows":
            if task == 0:
                cmd = r'copy {0} {1}'.format(os.path.join(S_Root, file_name), os.path.join(S_Destiny_Win, file_name))
                print(cmd)
                os.system(cmd)
            elif task == 1:
                cmd = r'copy {0} {1}'.format(os.path.join(F_Root, file_name), os.path.join(F_Destiny_Win, file_name))
                print(cmd)
                os.system(cmd)
            elif task == 2:
                cmd = r'copy {0} {1}'.format(os.path.join(P_Root, file_name), os.path.join(P_Destiny_Win, file_name))
                print(cmd)
                os.system(cmd)
        elif platform_name == "Linux":
            if task == 0:
                cmd = r'cp {0} {1}'.format(os.path.join(S_Root, file_name), os.path.join(S_Destiny_Lin, file_name))
                print(cmd)
                os.system(cmd)
            elif task == 1:
                cmd = r'cp {0} {1}'.format(os.path.join(F_Root, file_name), os.path.join(F_Destiny_Lin, file_name))
                print(cmd)
                os.system(cmd)
            elif task == 2:
                cmd = r'cp {0} {1}'.format(os.path.join(P_Root, file_name), os.path.join(P_Destiny_Lin, file_name))
                print(cmd)
                os.system(cmd)
    print("getProcessedData Done !")

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="financial", type=str, help="Select data for processing")
args = parser.parse_args()
if args.task == 'semeval':
    ProcessSemData()
    transferProcessedData(0)
elif args.task == 'financial':
    ProcessFinancialData()
    transferProcessedData(1)
elif args.task == 'people':
    ProcessPeopleData()
    transferProcessedData(2)

