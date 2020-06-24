#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: weights.py
#@time: 2020/6/24 9:37
#@email:chenbinkria@163.com
"""

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
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
weights = model.get_weights()
pass