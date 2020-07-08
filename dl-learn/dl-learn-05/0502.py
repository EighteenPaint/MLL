#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: 0502.py
#@time: 2020/6/22 10:00
#@email:chenbinkria@163.com
"""
from keras import layers
from keras import models

model = models.Sequential()
# 创建CNN
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
# 创建密集层
# add
model.add(layers.Dropout(0.5))  #
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# 编译模型
from keras import optimizers

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
# 准备数据集
train_dir = r'E:\dataset\dataset-dog-cat - small\training_set\training_set'
test_data = ''
validation_dir = r'E:\dataset\dataset-dog-cat - small\validation-set\validation-set'
from keras.preprocessing.image import ImageDataGenerator
# add rotation_range,width_shift_range,height_shift_range，shear_range，zoom_range，horizontal_flip 进行数据增强
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20,
                                                    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20,
                                                        class_mode='binary')

# 训练生成器
"""
steps_per_epoch=100：每个批次训练100个图片
epochs=30:训练30次，跟model.fit参数是一致的
validation_data=validation_generator：
validation_steps=50：
"""
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator,
                              validation_steps=50)
model.save('cat_and_dogs_small_1.h5')

# 可视化训练过程

import matplotlib.pyplot as plt

# 精确度数据
acc = history.history['acc']
val_acc = history.history['val_acc']
# 损失值
loss = history.history['loss']
val_loss = history.history['val_loss']
# 使用epoch作为x轴
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
