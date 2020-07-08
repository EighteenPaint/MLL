#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: 0603.py
#@time: 2020/6/24 10:19
#@email:chenbinkria@163.com
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = r'E:\dataset\jena_climate_2009_2016.csv\jena_climate_2009_2016.csv'
train_data = pd.read_csv(path).values[:, 1:].astype(float)  # 这里一定要转为浮点型，否则默认是object，在计算标准差时会出现问题
mean = train_data[:200000].mean(axis=0)
# 标准化数据
train_data = train_data - mean
# std = train_data[:200000].std(axis=0)
std = np.std(train_data[:200000], axis=0)  # 这里曾经因为类型不对应无法进行计算，保险起见最好转换一下或者检查一下
data = train_data / std


# print(data)


# 生成器
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    """

    :param data: 数据集
    :param lookback: 输入数据应该包含多少过去的数据集
    :param delay: 目标应该在未来多少个时间步之后
    :param min_index:用于界定需要抽取哪些时间步 与max_index连用
    :param max_index:用于界定需要抽取哪些时间步 与min_index连用
    :param shuffle:打乱样本还是顺序抽取样本
    :param batch_size:每个批量的样本数
    :param step:数据采样的周期（单位时间步），每个时间步是10分钟
    :return:
    """
    if max_index is None:
        max_index = len(data) - delay - 1  # 如果没有指定最大索引，就选用想要能够预测的最后一个，预测最后一个就需要delay前条数据
    i = min_index + lookback  # 需要过去lookback才能足够预测未来，所以 i = min_index + lookback，如果直接是min_index,如果前面没有lookback个数据，那就会出问题
    while True:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback,
                max_index,
                size=batch_size
            )
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets


lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(train_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(train_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(train_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(train_data) - 300001 - lookback) // batch_size

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, train_data.shape[-1])))  # 这里直接压平进行训练，没有使用CNN或者RNN，相当于直接使用线性规划
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')  # 这里使用标准差来衡量
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
