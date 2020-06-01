#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: DTree.py
#@time: 2020/5/31 21:07
"""

from math import log
import numpy as np


def calcShannnoEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    shannoEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannoEnt -= prob * log(prob, 2)
    return shannoEnt


def createDataSet():
    dataSet = np.array(
        [
            [1, 1, "Y"],
            [1, 0, "N"],
            [0, 1, "N"],
            [0, 1, "N"]

        ]
    )
    label = ["no surfaceing", "flippers"]
    return dataSet, label


def splitDataSet(dataSet, axis, value):
    """
    :param dataSet: 数据集
    :param axis: 分类属性所在的列
    :param value: 分类属性的值，用什么值来进行分类
    :return:
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis]  # 去掉特征的那个数据
            reduceFeatVec.extend(featVec[axis + 1:])  # 相当于从所选特征的数据那里砍一刀
            retDataSet.append(reduceFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 计算有多少个特征属性，因为最后一个不是属性而是类别的判定
    baseEntropy = calcShannnoEnt(dataSet)
    bestGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannnoEnt(subDataSet)  #此算法来自于统计学习方法和西瓜书里均有讲解
        infoGain = baseEntropy - newEntropy
        if infoGain > bestGain:
            bestFeature = i
            bestGain = infoGain
    return bestFeature