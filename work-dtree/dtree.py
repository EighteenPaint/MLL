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
            reduceFeatVec = list(featVec[:axis]) # 去掉特征的那个数据
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
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannnoEnt(subDataSet)  # 此算法来自于统计学习方法和西瓜书里均有讲解
        infoGain = baseEntropy - newEntropy
        if infoGain > bestGain:
            bestFeature = i
            bestGain = infoGain
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    keys = classList.keys()
    for vote in classList:
        if vote not in keys:
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda a: a[1], reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """

    :param dataSet: 数据集
    :param labels: 属性标签
    :return:
    """
    classList = [example[-1] for example in dataSet]
    """
    count():用于统计某个字符串出现的次数，这里表示类别出现的次数
    这里表示如果这个数据集都是一个类别的，那就不需要继续划分
    """
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:  # 能够用来划分的属性已经没有了，但是数据集还可以在分，此时可以选择投票机制
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 得到最好的特征属性，以便用来划分
    bestFeatLabel = labels[bestFeat]  # ？？？
    myTree = {bestFeatLabel: {}}  # 通过dict方式来构建tree
    del (labels[bestFeat])  # 用来分割的属性在子数据集中可以移除
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # 由于python的list是可变的，保险起见，新建一个变量
        myTree[bestFeatLabel][value] = createTree(splitDataSet(
            dataSet, bestFeat, value
        ), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]  # 树根
    secondDict = inputTree[firstStr]  # 子树
    featIndex = featLabels.index(firstStr)  # 找到树根标签所对应的属性索引值
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == "dict":  # 如果还有子树，说明继续进行决策
                classLabel = classify(secondDict, featLabels, testVec)  # 这里进行了递归
            else:
                classLabel = secondDict[key]  # 没有子树的话，说明已经搜索完毕
    return classLabel
