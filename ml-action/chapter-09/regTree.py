#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: regTree.py
#@time: 2020/6/8 9:42
#@email:chenbinkria@163.com
"""

import numpy as np


def loadDataSet(fiilName):
    dataMat = []
    with open(fiilName) as file:
        for line in file.readlines():
            curline = line.strip().split('\t')
            fltLine = list(map(float, curline))  # 把curline的每一个元素转化成浮点型
            dataMat.append(fltLine)
        return dataMat


def binSplitDataSet(dataSet, feature, value):
    """
    数据集切割
    :param dataSet:
    :param feature:
    :param value:
    :return:
    """
    """
    dataSet[:, feature] > value ====>[1,1,1,0,0,0],数据集的每一行的feature会跟value做比较，返回并返回相应的True or FALSE，形成一个数组
    数组的索引就是数据集的行号
    于是代码就变成：np.nonzero([1,1,1,0,0,0])  ====>   ([0,0,0],[0,1,2]):表示第0行第0列,第0行，第1列，第0行第2列元素为非零元素,实际上，当数组是一维的时候，返回的
    是（[0,1,2],意思是：取0行，1行，2行的数据
    """
    mat0 = []
    mat1 = []
    lineIndex = np.nonzero(dataSet[:, feature] > value)[0]
    if len(lineIndex) != 0:
        mat0 = dataSet[lineIndex, :]  # ???
    lineIndex = np.nonzero(dataSet[:, feature] <= value)[0]
    if len(lineIndex) != 0:
        mat1 = dataSet[lineIndex, :]
    return mat0, mat1


def regLeaf(dataSet):
    """
    负责生产叶子节点,在回归树中我们使用均值作为回归的树的叶子节点
    :param dataSet:
    :return:
    """
    return np.mean(dataSet[:, -1])


def regError(dataSet):
    """
    采用总方差来衡量误差
    :param dataSet:
    :return:
    """
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def createTree(dataSet, leafType=regLeaf, errType=regError, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val
    retTree = {'sInd': feat, 'spVal': val}
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regError, ops=(1, 4)):
    """

    :param dataSet:训练的数据集
    :param leafType:叶子节点创建
    :param errType: 计算误差的方式，回归树采用总误差方式
    :param ops: ops_1:容许的误差下降值；ops_2:切分的最少样本数
    :return:
    """
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 如果只有一个特征值，则只有一个叶子节点
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)  # 所有样本的方差
    bestS = float('inf')  # 最好的方差，初始化为无穷
    bestIndex = 0
    bestValue = 0
    for festIndex in range(n - 1):  # 遍历所有的属性
        for splitVal in set(dataSet[:, festIndex].T.tolist()[0]):  # 遍历属性的每一个值，尝试进行划分并计算误差
            mat0, mat1 = binSplitDataSet(dataSet, festIndex,
                                         splitVal)  # 对每一个feature value都进行一次划分，然后找到最好的feature index and feature value
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # 如果划分的数据集只有不多于tolN，就不需要进行后面的步骤
                continue
            newS = errType(mat0) + errType(mat1)  # 计算总的误差和
            if newS < bestS:
                bestIndex = festIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:  # ？？？？
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    return (bestIndex, bestValue)
