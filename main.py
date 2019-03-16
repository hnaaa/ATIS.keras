# -*- coding:utf-8 -*-
import numpy as np
import pickle

import data.load
from metrics.accuracy import conlleval
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv1D, MaxPooling1D
import os

import progressbar


### Load Data
train_set, valid_set, dicts = data.load.atisfull()   #从atis.pkl中载入atis数据集。分别为训练集验证集和字典
w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']#w2idx是 词汇到索引的字典，ne2idx不需要，labels2idx是标签(槽)到索引的词典

# Create index to word/label dicts
idx2w  = {w2idx[k]:k for k in w2idx}        #索引到词汇 类推
idx2ne = {ne2idx[k]:k for k in ne2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}


### Model
n_classes = len(idx2la)   #词的分类个数(槽)127
n_vocab = len(idx2w)   #词表大小 572
# print(n_classes,n_vocab)

# Define model
model = Sequential()
model.add(Embedding(n_vocab,100))               #将词汇索引转化为100维的word vec
# input (batch,词汇个数，词表大小)（onehot）
# output(batch，词汇个数，100) word_vec
model.add(Conv1D(64,5,padding='same', activation='relu'))#对输入词向量进行一维卷积提取n-gram信息  64个卷积核，卷积核大小为5*100
#input (batch,词汇个数，100)
#output (batch,词汇个数，64)
model.add(Dropout(0.25)) #略
model.add(GRU(100,return_sequences=True)) #将提取的词汇信息通过GRU进一步提取结构信息和时序信息 输出维度为100维
#input (batch,词汇个数，64)
#output (batch,词汇个数，100)
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))#时序分配的全连接层，将全连接作用到每一个GRU的输出上
#input (batch,词汇个数，100)
#output (batch，词汇个数，127)
model.compile('rmsprop', 'categorical_crossentropy') #rmsprop优化方法 crossentropy损失函数
model.summary()

### Ground truths etc for conlleval
train_x, train_ne, train_label = train_set    #将整个训练集顺序解包乘 训练数据train_x的索引形式，train_ne没有用到，train_label 标签的索引形式 比如 1 3 5 77 9
val_x, val_ne, val_label = valid_set  #同理

words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]          #验证集数据的真实词汇形式  比如 I from Canada
groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]  #验证集标签的真实词汇形式
words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]  #类推
groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]


### Training
n_epochs = 100

train_f_scores = []
val_f_scores = []
best_val_f1 = 0
#以下为训练和验证过程 用到了conlleval脚本计算f1值，f1 = (P+R)/2*P*R P是精确率 R是召回率
for i in range(n_epochs):
    print("Epoch {}".format(i))

    print("Training =>")
    train_pred_label = []
    avgLoss = 0

    bar = progressbar.ProgressBar(max_value=len(train_x))
    for n_batch, sent in bar(enumerate(train_x)):
        label = train_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]

        if sent.shape[1] > 1: #some bug in keras
            loss = model.train_on_batch(sent, label)
            avgLoss += loss

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        train_pred_label.append(pred)

    avgLoss = avgLoss/n_batch

    predword_train = [ list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
    con_dict = conlleval(predword_train, groundtruth_train, words_train, 'r.txt')
    train_f_scores.append(con_dict['f1'])
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))


    print("Validating =>")

    val_pred_label = []
    avgLoss = 0

    bar = progressbar.ProgressBar(max_value=len(val_x))
    for n_batch, sent in bar(enumerate(val_x)):
        label = val_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]

        if sent.shape[1] > 1: #some bug in keras
            loss = model.test_on_batch(sent, label)
            avgLoss += loss

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        val_pred_label.append(pred)

    avgLoss = avgLoss/n_batch

    predword_val = [ list(map(lambda x: idx2la[x], y)) for y in val_pred_label]
    con_dict = conlleval(predword_val, groundtruth_val, words_val, 'r.txt')
    val_f_scores.append(con_dict['f1'])

    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))

    if con_dict['f1'] > best_val_f1:
    	best_val_f1 = con_dict['f1']
    	open('model_architecture.json','w').write(model.to_json())
    	model.save_weights('best_model_weights.h5',overwrite=True)
    	print("Best validation F1 score = {}".format(best_val_f1))
    print()

