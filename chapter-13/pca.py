#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: pca.py
#@time: 2020/6/13 9:36
#@email:chenbinkria@163.com
"""

import numpy as np


def loadDataSet(fileName, delim='\t'):
    with open(fileName) as fr:
        stringArr = [line.split(delim) for line in fr.readlines()]
        dataArr = list(map(lambda d: list(map(float, d)), stringArr))
        return np.mat(dataArr)


def pca(dataMat, topNfeat=999999):
    meanVals = np.mean(dataMat, axis=0)  # 计算平均数
    meanRemoved = dataMat - meanVals  # 数据去中心化处理
    covMat = np.cov(meanRemoved, rowvar=0)  # 计算协方差
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 计算特征值和特征向量
    eigValInd = np.argsort(eigVals)  # 对特征值进行排序
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 返回前topN个特征
    redEigVects = eigVects[:, eigValInd]  # 返回相应的特征向量
    lowDDataMat = meanRemoved * redEigVects  # 进行空间转换
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  # 逆变换
    return lowDDataMat, reconMat
