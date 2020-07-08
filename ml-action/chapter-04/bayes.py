#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: bayes.py
#@time: 2020/6/1 16:11
"""

import numpy as np
from math import log


def loadDataSet():
    """
    下面的数据可以这样看：classVec相当于标签列，每一个数字对应
    [                             文本                               ]      [类别]
    ['my',        'dog', 'has', 'flea', 'problems', 'help', 'please']       [0]
    ['maybe',    'not', 'take', 'him', 'to', 'dog', 'park', 'stupid']       [1]
    ['my',       'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him']       [0]
    ['stop',             'posting', 'stupid', 'worthless', 'garbage']       [1]
    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him']       [0]
    ['quit',          'buying', 'worthless', 'dog', 'food', 'stupid']       [1]
    :return:
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 求并集
    return list(vocabSet)


def setOfWord2Vec(vocalList, inputSet):
    returnVec = [0] * len(vocalList)  # 一个全为零的向量，长度跟vocalList一样
    for word in inputSet:
        if word in vocalList:
            returnVec[vocalList.index(word)] = 1
        else:
            print(f"the word {word} is not in my Vocabulary!")
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """

    :param trainMatrix: 一行一行的单词
    :param trainCategory: 每篇文档对应的类别所构成的向量
    :return:
    """
    """
    trainMatrix:把每一行数据都向量化,是一个n维矩阵
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 
        [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], 
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    ]
    trainCategory:来自于初试数据集中
    [1,1,1,1,1,1]
    """
    numTrainDocs = len(trainMatrix)  # 获取有多少行数据
    numWordsOfLine = len(trainMatrix[0])  # 每一行的向量
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 侮辱性词汇与所有词汇占比，也就是P(ci)
    p0Num = np.ones(numWordsOfLine)  # 如果有太多的的0会导致最后的结果为0与，使用1作为初始值
    p1Num = np.ones(numWordsOfLine)
    p0Denom = p1Denom = 2.0  # 这里涉及到去零的处理，应该会有更好的做法，比如往右所说的拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  # 如果是侮辱性词汇
            p1Num += trainMatrix[i]  # 单词出现一次就+1
            p1Denom += sum(trainMatrix[i])  # 累计侮辱性的句子中，出现的所有词
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])  # 在非侮辱性句子中，侮辱性词汇的总和
    p1Vect = log(p1Num / p1Denom)  # p([wi]|1):在侮辱性词汇中，单词的出现概率
    p0vect = log(p0Num / p0Denom)  # p([wi]|0):在非侮辱性词汇中，单词的出现概率
    return p0vect, p1Vect, pAbusive


def classfy(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    :param vec2Classify: 待预测向量
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    """
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)   #由于取对数，将原来的相乘转换为相乘
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)   #这里我们会问，p(w)的值呢，p（w），因为比较大小，而且p(w)一样，那不对
    if p1 > p0:
        return 1
    else:
        return 0
