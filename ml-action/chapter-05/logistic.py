#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: logistic.py
#@time: 2020/6/2 15:48
"""
import numpy as np


def loadDataSet():
    """
    b   x0   x1
    -----------
    1   2    3
    1   2    3
    1   2    3
    1   2    3
    1   2    3
    1   2    3
    1   2    3
    :return:
    """
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # 1.0相当于常数b
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()  # 矩阵转置
    m, n = np.shape(dataMatrix)
    alpha = 0.01  # 这也是一个不好确定的数值，很多时候也许要尝试很多次才可以得出来
    maxCycle = 500  # 循环次数500
    """
    dataMatrix * weight
    1   2    3
    1   2    3
    1   2    3  1
    1   2    3  1 
    1   2    3  1
    1   2    3
    1   2    3    
    """
    weights = np.ones((n, 1))
    for k in range(maxCycle):
        h = sigmoid(dataMatrix * weights)  # 矩阵运算
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error  # 这种方法处理妙啊，明天推导一下,为什么要乘以error
    return weights

