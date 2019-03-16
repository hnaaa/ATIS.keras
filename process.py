# -*-coding:utf-8-*-

'''
data precess module
by hn
'''
import os
from pyhanlp import *
def list_all_file(inputdir):
    '递归返回目录下的所有文件名称'
    assert os.path.isdir(inputdir),'输入的不是目录'
    for path,dirnames,files in os.walk(inputdir):
        if dirnames:
            for dir in dirnames:
                list_all_file(os.path.join(path,dir))
        if files:
            for file in files:
                yield os.path.join(path,file)




def buildtable(input_dir,output_file):
    '''
    输入一个文件夹名称
    将其中每一个分好词的文本文档中的所有词中，检测出与输出文件不重复的词语，并且标注索引
    '''
    '''参数:input_dir:输入文件夹名称
            output_file:输出名称名称
    '''
    assert os.path.isdir(input_dir) ,'input not exist'
    old_words = []
    new_words=[]
    idx_off = 0
    with open(output_file,'r') as output_file_handle:
        while 1:
            line = output_file_handle.readline()
            if not line:
                break
            idx_off += 1
            old_words.append(line.split()[0])
    for file in list_all_file(input_dir):
        with open(file, 'r') as input_file_handle:
            while 1:
                line = input_file_handle.readline()
                if not line:
                    break
                words = line.split()
                for word in words:
                    if word not in old_words:
                        old_words.append(word)
                        new_words.append(word)
    with open(output_file, 'a') as output_file_handle:
         for index,word in enumerate(new_words):
             output_file_handle.writelines([word,' ',str(index+idx_off),'\n'])
def seged_words_to_idx(input_path,voc_path,output_path):
    '将输入的分好词文件按行读取后，每行转化成一个索引序列'
    '''
    参数:input_path输入的文件路径
    voc_path 词表的路径
    output_path 输出文件的路径
    '''
    w2idx = {}
    idxs_all = []
    assert os.path.exists(input_path) and os.path.exists(voc_path) ,'词表或者输入文件有误'
    with open(voc_path,'r') as file_handle:
        while 1:
            line = file_handle.readline()
            if not line:
                break
            word,idx = line.split()
            w2idx[word] = idx
    with open(input_path,'r') as file_handle:
        while 1:
            words = file_handle.readline().split()
            if not words:
                break
            idxs = [w2idx[i] for i in words]
            idxs_all.append(idxs)
    with open(output_path,'w') as file_handle:
        for idxs in idxs_all:
            str_idx = ''
            for idx in idxs:
                str_idx += idx
                str_idx += ' '
            file_handle.writelines([str_idx,'\n'])

def dir_seg(input_dir,output_dir):
    '将输入文件夹中的所有文件分词后放入输出文件夹'
    assert os.path.isdir(input_dir) and os.path.isdir(output_dir),'输入文件夹或输出文件夹错误'
    for file in list_all_file(input_dir):
        print(file)
        output_name = os.path.join(output_dir,os.path.split(file)[1]).replace('.txt','_seg.txt')
        print(output_name)
        file_seg(file,output_name)

def file_seg(input_file,output_file):
    '将输入的文本转换为分词后的文本'
    '''
    参数...
    '''
    words_all = []
    assert os.path.exists(input_file),'输入文件不存在！'
    with open(input_file,'r') as file_handle:
        while 1:
            line = file_handle.readline()
            if not line:
                break
            line = line.replace('\n','')
            words_all.append(line)
    with open(output_file,'w') as file_handle:
        for words in words_all:
            line_str = []
            seged_words = HanLP.segment(words)
            for item in seged_words:
                if str.isdigit(item.word):
                    line_str.append('DIGIT')
                else:
                    line_str .append(item.word)
            line_str = ' '.join(line_str)
            file_handle.write(line_str+'\n')
def words_to_indexs(input_file,output_file):
    '直接将原始句子文档转化为分好词的索引句子文档'
    pass











if __name__ == '__main__':
    help(file_seg)

