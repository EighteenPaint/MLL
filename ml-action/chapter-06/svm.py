#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: svm.py
#@time: 2020/6/4 9:05
#@email:chenbinkria@163.com
"""
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat


def selectJrand(i, m):
    """
    只要函数输入值不等于i，函数就会进行随机选择
    :param i: α的下标
    :param m: 所有α的数目
    :return:
    """
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    """

    :param dataMatIn: 数据集
    :param classLabels: 类别标签
    :param C: 常数C 注意这个C值是我们自己指定的数字
    :param toler:容错率  例子中用的是0.001
    :param maxIter:最大迭代次数
    :return:
    """
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    alphas = np.mat(np.zeros((m, 1)))  # 阿尔法初始化为0
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):  # 有m行数据，所以遍历m次
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b  # 由于经过处理，f(x)可以转化为由α表示的式子
            Ei = fXi - float(labelMat[i])  # if checks if an example violates KKT conditions，
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):  # 判断是否满足KKT条件，具体的证明可见统计学习方法具有详细的
                j = selectJrand(i, m)  # 进行内循环，处理α2，这里是随机选的
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):  # 这个也是可以证明的，可以直接照搬
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H");
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T \
                      - dataMatrix[i, :] * dataMatrix[i, :].T \
                      - dataMatrix[j, :] * dataMatrix[j, :].T  # 求解α用
                if eta >= 0:
                    print("eta>=0");
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)  # 就是一次更新操作
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("j not moving enough");
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])  # update i by the same amount as j  #  α2确定之后，α1也就确定了
                # the update is in the oppostie direction
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - labelMat[
                    j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T  # 求解b值得方式，证明可见《统计学习方法》
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1  #进行了更新，该属性也要相应改变
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas
