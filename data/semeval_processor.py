import os
import collections

class SemEvalProcessor():
    '''Process the SemEval-2010 dataset.'''

    
    def __init__(self):
        super().__init__()
    
    def relation_2_id(self, filename, output_filename):
        '''
        Map the relation to corresponding ID, 
        then save the map in file and return a ordered dict.
        @param
            filename: Target file remains to be processed.
            output_filename: The file that stores the mapped dict.
                             The dict's format is: (relation - id).  
        '''
        relation = collections.OrderedDict()
        id = 0
        # Read training file to get all relations.
        with open(filename, 'r') as f1:
            for idx, line in enumerate(f1.readlines()):
                if idx % 4 == 1:
                    if line not in relation:
                        relation[line] = id 
                        id += 1
        
        # Store the mapping dict. The relation and its ID is seperated by a space.
        print(relation)
        with open(output_filename, 'w') as f2:
            for k, v in relation.items():
                f2.write(k.replace('\n', '') + " " + str(v) + '\n')
        print("Relation_2_id Done !")
        return relation

    # 我觉得可以删了，通过list(relation.keys())来获取即可
    def get_labels(self, input_file, output_file):
        '''
        Get all the labels.
        '''

        relations = []
        with open(input_file, 'r', encoding='utf-8') as f: 
            for line in f.readlines():
                relation = line.split(" ")[0].strip()
                relations.append(relation)
        with open(output_file, 'w', encoding='utf-8') as f: 
            for data in relations:
                f.write(data + '\n')
        print("Label File Done !")

    
    def _add_spaces(self, sentence):
        """对于每个句子，在<e1>, </e1>, <e2>, </e2>前后分别加上空格以区分
            该方法会用在format_process方法中
        Params:
            sentence : 待处理的句子
        """
        idx = [0, 0, 0, 0]    # idx保存各个<e>的位置下表，一共四个元素，初始化全为0
        idx[0] = sentence.find("<e1>") + 4
        idx[1] = sentence.find("</e1>") + 1
        idx[2] = sentence.find("<e2>") + 6
        idx[3] = sentence.find("</e2>") + 3 
        sentence_list = list(sentence)
        for i in range(4):
            sentence_list.insert(idx[i], ' ')        
        return ''.join(sentence_list)

    def format_process(self, filename, output_sentences_filename, 
                        output_relations_filename):
        """将给定格式的文件中的句子单独分出来，用作训练前的准备
        Params:
            filename : 待处理的文件名
            output_sentences_filename : 保存句子的文件名
            output_relations_filename : 保存每条句子对应关系的文件名
            relation_2_id_filename :    关系映射字典的文件名
        """
        sentences = []
        relations = []
        with open(filename) as f1:
            for _, line in enumerate(f1.readlines()):
                # 下标能够被4整除的是句子，紧挨句子的下一句是其对应的关系
                # 对于每一个句子，去除其开头的编号、空格以及首尾的双引号
                # 对句子剩余部分的实体分隔符进行前后添加空格操作，保存最终结果
                if _ % 4 == 0:
                    idx = line.find('\t')
                    sentences.append(self._add_spaces(line[idx + 2: -2]))
                if _ % 4 == 1:    
                    relations.append(line)
        # 保存
        with open(output_sentences_filename, 'w') as f2:
            for sentence in sentences:
                f2.write(sentence+'\n')
        with open(output_relations_filename, 'w') as f3:
            # f4 = open(relation_2_id_filename, 'r')
            # relation2id = f4.readlines()
            # f4.close()
            for relation in relations:
                f3.writelines(relation)
        print("Format_process Done !")
    
    def combine_sentence_relation(self, sentence_filename, relation_filename, output_filename):
        """将每条句子与其对应的关系组合成结构化文件
        Params:
            sentence_filename: 待处理的句子文件
            relation_filename: 对应的关系文件
            output_filename: 保存结果的路径
        """
        with open(sentence_filename, 'r') as f1:
            sentences = f1.readlines()
        with open(relation_filename, 'r') as f2:
            relations = f2.readlines()
        # 若句子列表长度与关系列表长度不相等，说明对应关系不匹配，抛出异常
        assert len(sentences) == len(relations)
        with open(output_filename, 'w') as f3:
            for i in range(len(sentences)):
                f3.writelines(relations[i].replace('\n', '') + '\t' + sentences[i])
        print('Combination Done !')        
