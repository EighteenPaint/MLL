#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: KNN.py
#@time: 2020/5/30 22:55
"""
from os import listdir

import numpy as np


def createDataSet():
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    label = ["A", "A", "B", "B"]
    return group, label


def classfy(inX, dataSet, ladels, k):
    """
    This is a function that could reuse
    :param inX: your data
    :param dataSet: trained data set
    :param ladels: class
    :param k: top k that min
    :return: classfied label
    """
    dataSetSize = dataSet.shape[0]  # shape is tuple，means its dim，shape[0] means row,shape[1] means column,now it is 4
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet  # tile api of numpy
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)  # 沿着横轴进行求和
    distance = sqDistance ** 0.5
    sortedDistIndicies = distance.argsort()  # eg.a[0] = 1表示待测试数据与标准数据集第一个数据的距离，获取的索引就是就是相应的标准
    classCount = {}  # 定义一个字典，将类别作为Key，便于计数
    for i in range(k):
        index = sortedDistIndicies[i]
        votelabel = ladels[index]
        classCount[votelabel] = classCount.get(votelabel, 0) + 1
    clazz = sorted(classCount.items(), key=lambda a: a[1], reverse=True)
    return clazz[0][0]


def createPersonDataSet(fileName: str):
    with open(fileName) as dataFile:
        lines = dataFile.readlines()
        rows = len(lines)
        column = 3
        mat = np.zeros((rows, column))
        index = 0
        labels = []
        for line in lines:
            line = line.strip()
            listFromLine = line.split("\t")
            mat[index, :] = listFromLine[0:3]  #
            labels.append(int(listFromLine[-1]))
            index += 1
        return mat, labels


def autoNorm(dataSet):
    minVal = dataSet.min(0)  # 这是np数组运算的强大之处，很多运算都可以以数组为单位
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    normDataSet = np.zeros(shape=np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVal, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVal


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),  key=lambda a: a[1], reverse=True)
    return sortedClassCount[0][0]


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # load the training set
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))
