#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/5/30 22:25
"""
import numpy as np

import KNN
import matplotlib.pyplot as plt

if __name__ == '__main__':
    group, label = KNN.createPersonDataSet("person.data")
    norm, *a = KNN.autoNorm(group)
    KNN.classfy()
    fig = plt.figure()  # 定义一张大的画布
    ax = fig.add_subplot(111)  # 添加一个子图，111等价于add_subplot(1,1,1)
    ax.scatter(norm[:, 1], norm[:, 0], 15.0*np.array(label), 15.0*np.array(label)) #取第一列数据和第二列数据
    plt.show()

    # print(norm)
