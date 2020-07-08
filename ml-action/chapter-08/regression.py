#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: regression.py
#@time: 2020/6/7 11:23
#@email:chenbinkria@163.com
"""
import numpy as  np


def loadDataSet(fileName):
    with open(fileName) as file:
        dataMat = []
        labelMat = []
        for line in file.readlines():
            lineArr = []
            curLine = line.strip().split('\t')
            for data in curLine:
                lineArr.append(float(data))  # transform tuple to array
            dataMat.append(lineArr)
            labelMat.append(float(curLine[-1]))
        return dataMat, labelMat


def standRegres(xArr, yArr):
    """
    这是一个最简单的线性回归
    :param xArr:
    :param yArr:
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:  # 计算行列式的值
        print("矩阵不可逆,已终止")
        return
    ws = xTx.I * (xMat.T * yMat)  # .I 表示取逆矩阵,该公式通过求导得到
    return ws


def lwlr(testPoint, xArr, yArr, k=1.0):
    """

    :param testPoint:
    :param xArr:
    :param yArr:
    :param k: 其大小会造成欠拟合或者过拟合，当你越小时，越容易过拟合
    :return:
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))  # 创建一个对角矩阵
    for j in range(m):  # 每一个待测试数据都要跟所有的数据集去计算weights
        diffMat = testPoint - xMat[j, :]  # 书上的公式是绝对值，但是我们计算的时候，绝对值不好处理，往往会用平方来做
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        return
    ws = xTx.T * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yhat = np.zeros(m)
    for i in range(m):
        yhat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yhat


def ridgeRegres(xMat, yMat, lam=0.2):
    """
    高纬度小样本容易出现过拟合
    :param xMat:
    :param yMat:
    :param lam:
    :return:
    """
    xTx = xMat.T * xMat
    m, n = np.shape(xMat)
    demon = xTx + np.eye(n) * lam
    if np.linalg.det(demon) == 0.0:
        return
    ws = demon.I * (xMat.T * yMat)


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMean = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)  # 方差
    xMat = (xMat - xMean) / xVar  # 将数据标准化，建议作为必要手段之一，可以很好地额
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat
