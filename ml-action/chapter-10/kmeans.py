#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: kmeans.py
#@time: 2020/6/9 10:15
#@email:chenbinkria@163.com
"""
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as file:
        for line in file.readlines():
            curline = line.strip().split('\t')
            fltLine = list(map(float, curline))  # python3 之后需要手动转换成list
            dataMat.append(fltLine)
        return dataMat


def distEclud(vecA, vecB):
    """
    距离平方和
    :param vecA:
    :param vecB:
    :return:
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # 平方和


def randCent(dataSet, k):
    """

    :param dataSet:
    :param k:
    :return:
    """
    n = np.shape(dataSet)[1]  # 获取维度信息
    centroid = np.mat(np.zeros((k, n)))  # k个n维向量，即K的质心
    for j in range(n):
        """
        为了保证随机选取的质心在边界之内，所以选择在最大值和最小值之间
        """
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(dataSet[:, j]) - minJ)
        centroid[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroid


def kMeans(dataSet, k, distMeas=distEclud, createCenter=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  #???
    centroids = createCenter(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = float("inf")
            minIndex = -1
            """
            这个循环是为了找到离dataSet[i, :]最近的点
            """
            for j in range(k):  #对每个质心进行计算，找到一个点的最近质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            """
            保存每个样本数据的最近质心点和距离的平方
            """
            if clusterAssment[i, 0] != minIndex:    # 这里就是收敛条件，当收敛时，minIndex不会在改变，只要有一个还在改变，说明还没有完全收敛
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        """
        更新k得值
        """
        for cen in range(k):
            ptsInClust = dataSet(np.nozero(clusterAssment[:, 0].A == cen)[0])  # 取出所有与第K个质心点关联的距离
            centroids[cen, :] = np.mean(ptsInClust, axis=0)  # 然后求均值
    return centroids, clusterAssment
