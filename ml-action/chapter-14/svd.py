#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: svd.py
#@time: 2020/6/15 16:09
#@email:chenbinkria@163.com
"""

import numpy as np
from numpy import linalg as la


def ecludSim(inA, inB):
    """
    基于欧式距离的相似度计算
    :param inA:
    :param inB:
    :return:
    """
    return 1.0 / (1.0 + la.norm(inA - inB))  # 求范数，实际上就是我们说的欧式距离


def pearsSim(inA, inB):
    """
    皮尔森系数求解
    :param inA:
    :param inB:
    :return:
    """
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    """
    余弦相似度
    :param inA:
    :param inB:
    :return:
    """
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simToatal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overlap = np.nonzero(np.logical_and(
            dataMat[:, item].A > 0,
            dataMat[:, j].A > 0
        ))[0]  # 找到所有对item 和 j 都评了级的用户
        if len(overlap) == 0:
            similarity = 0
        else:
            similarity = simMeas(
                dataMat[overlap, item],
                dataMat[overlap, j]
            )  # 计算两个item的相似度
        simToatal += similarity  # 把所有的相似度加起来
        ratSimTotal += similarity * userRating  # 这里相似度相当于权重了，我对A的品级五星，B跟A 90%相似，那就对B的评级就是5*0.9 = 4.5
    if simToatal == 0:
        return 0
    else:
        return ratSimTotal / simToatal  # 这里类似于求一个平均值，归一化操作，这种归一化操作的方式值得思考


def recommand(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return "you rated everything"
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda a: a[1], reverse=True)[:N]


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def loadExData3():
    return np.mat([[4, 4, 0, 2, 2],
            [4, 0, 0, 3, 3],
            [4, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]])


def loadExData2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
            [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
            [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
            [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
            [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
            [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
            [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
            [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
            [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]
