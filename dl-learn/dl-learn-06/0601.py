#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: 0601.py
#@time: 2020/6/22 22:14
#@email:chenbinkria@163.com
"""
from keras.preprocessing.text import Tokenizer
import numpy as np

smaple = ['The cat sat on my mat', 'The dog ate my homework']
# 使用numpy进行one hot 编码
'''
单词级编码即是对一个单词一个向量
句子级编码即是对一个句子一个向量
字符级编码即是对一个字符一个向量
书中在讲解one hot 编码时有点没有清楚，在使用numpy进行编码时其实是对每个单词进行编码，使用keras提供的编码工具，是对每个句子进行的编码
'''
token_index = {}
for sample in smaple:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
results = np.zeros(shape=(len(smaple), 10, max(token_index.values()) + 1))
for i, sample in enumerate(smaple):
    for j, word in list(enumerate(sample.split()))[:10]:
        index = token_index.get(word)
        results[i, j, index] = 1.
# print(results)

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(smaple)
seq = tokenizer.texts_to_sequences(smaple)  # 这里对每个单词出现频次从大到小进行了排序，所以这里产生的索引与我们自己实现的索引不一样
one_hot = tokenizer.texts_to_matrix(smaple, mode='binary')
# 嵌入词向量
from keras.layers import Embedding

# embedding_layer = Embedding(1000, 64)

from keras.datasets import imdb
from keras import preprocessing

max_features = 10000
maxlen = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))  # 输入10000条数据，每条数据长度是20个单词，需要嵌入到一个8维词向量空间
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
print(model.get_weights())
