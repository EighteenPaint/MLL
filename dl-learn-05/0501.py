#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/21 21:34
#@email:chenbinkria@163.com
"""

from keras import layers
from keras import models

# 创建卷积神经网络
model = models.Sequential()
# 卷积神经网络有固定的接收向量方式：（image_height,image_width,channel）=(高，宽，颜色通道），
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 创建密集连接网络,也就是说前面的过程单纯是为了处理图像
model.add(layers.Flatten())  # 将3D 铺平为1D
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 输出10个类别的相近程度
# model.summary()

# 导入数据
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))  # 这里单纯从张量角度确实不是很明白为何这样，我认为可以这样理解：CNN
# 他就是需要这样形状的数据，不用太纠结。因为不影响结果，所以按照CNN需要的进行变化即可
train_images = train_images.astype('float32') / 255  # 标准化，因为是灰度图，不需要这么高的数值，而且神经网络需要标准化
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, 64, 5)

# 测试模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_loss, test_acc)
