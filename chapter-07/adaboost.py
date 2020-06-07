#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: adaboost.py
#@time: 2020/6/6 10:08
#@email:chenbinkria@163.com
"""
import numpy as np


def loadSimpData():
    dataMat = np.mat([
        [1.0, 2.1],
        [2.0, 1.1],
        [1.3, 1.1],
        [1.0, 1.0],
        [2.0, 1.0]
    ])
    classLabel = [1.0, 1.0, -1.0, -1.0, -1.0]
    return dataMat, classLabel


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    训练决策树
    :param dataArr: 数据
    :param classLabels: 数据标签
    :param D: 每个数据的权重
    :return:
      :bestStump:训练好的分类器
      ：minError:错误率
      ：bestClasEst：训练的结果
    """
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = float("inf")
    for i in range(n):  # 通过对每个属性的都进行划分找到最好的划分属性：错误率最低的
        """
        获取最大值和最小值是为了获取一个切平面空间，结合步长就可以不断地进行尝试
        """
        rangeMin = dataMatrix[:, i].min()  # 第i列的最小值
        rangeMax = dataMatrix[:, i].max()  # 第i列最大值
        stepSize = (rangeMax - rangeMin) / numSteps  # 计算步长
        for j in range(-1, int(numSteps) + 1):
            """
            考虑到每一个属性都会有两种分法，
            比如：小于1的值是A类，大于1的是B类，也可以小于A的是B类，
            大于1的是A类，错误率是不一样的
            """
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0  # numpy的数组操作，中括号内可以直接判断
                weightedError = D.T * errArr  # 将错误也权重化
                """
                ???:为什么这里的错误率的计算跟提供的计算公式不一样
                """
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictedVals.copy()
                    """
                    保存训练好的分类器
                    """
                    bestStump['dim'] = i  # 单层决策树选择分类的属性
                    bestStump['thresh'] = threshVal  # 划分的切面值
                    bestStump['ineq'] = inequal  # 划分的方式
        return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, classLabel, numIt=40):
    """
    训练N个单层决策树，最多40个单层决策树，当然也可以选择多种不同的分类器进行训练
    :param dataArr:
    :param classLabel:
    :param numIt:
    :return:
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabel, D)
        print("D:", D.T)
        alpha = float(0.5 * np.log((1 - error) / np.max([error, 1e-16])))  # 为了防止出现error为零的情况
        bestStump['alpha'] = alpha  # 对一个分类器来说不仅要保持自己的参数，还需要保存其权重值
        weakClassArr.append(bestStump)  # 放入adaboost分类器数组里面
        print("ClassEst:", classEst)
        expon = np.multiply(-1 * alpha * np.mat(classLabel).T, classEst)
        D = np.multiply(D, np.exp(expon))  # 更新D值
        D = D / D.sum()
        aggClassEst += alpha * classEst  # ∑ αi * h(i)
        print("aggClassEst: ", aggClassEst.T)
        """
        两者进行的是数组的运算，这里要区分numpy的矩阵乘法数组乘法，数组的乘法需要通过方法来实现，而矩阵可以直接使用*符号，这也是在编码过程中需要区分的地方
        """
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabel).T, np.ones((m, 1)))
        errorsRate = aggErrors.sum() / m
        print("total erroes：", errorsRate)
        if errorsRate == 0.0:
            break
    return weakClassArr
