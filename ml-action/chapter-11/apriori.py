#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: apriori.py
#@time: 2020/6/9 16:14
#@email:chenbinkria@163.com
"""
import numpy as np


def loadDataSet():
    return [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]


def createC1(dataSet):
    """
    创建商品集
        不言自名，函数createC1()将构建集合C1。 C1是大小为1的所有候选项集的集合。 Apriori
    算法首先构建集合C1，然后扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要
    求。那些满足最低要求的项集构成集合L1。而L1中的元素相互组合构成C2， C2再进一步过滤变
    为L2。到这里，我想读者应该明白了该算法的主要思路
    :param dataSet:
    :return:
    """
    C1 = []
    for transaction in dataSet:  # 遍历每一个订单
        for item in transaction:  # 遍历订单中的每一项
            if not [item] in C1:  # 如果这一项不在C1里面，就加入
                C1.append([item])
    C1.sort()
    """
    Python3.X map返回类型不再是list，需要手动转换
    """
    return list(map(frozenset, C1))


def ScanD(D, Ck, minSupport):
    """
     扫描数据集来判断这些只有一个元素的项集是否满足最小支持度的要
    求。那些满足最低要求的项集构成集合L1。而L1中的元素相互组合构成C2， C2再进一步过滤变
    为L2。
    :param D:
    :param Ck:
    :param minSupport:
    :return:
    """
    ssCnt = {}
    for tid in D:
        for can in Ck:
            """
            如果希望IDE可以提示，可以指定类型，比如Ck: set[set]
            """
            if can.issubset(tid):  # 判断商品是否在订单中出现，若是子集，则出现一次
                if can not in ssCnt:  # if not ssCnt.has_key(can):  # python3 之后不会再有has_key
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData


def aprioriGen(Ln, k):
    """
    :param Ln:频繁集
    :param k:项集元素个数,只可以比Lk频繁集项目个数大1
    :return:项集
    :note: k = length(Ln[i]) + 1,eg：如果Ln的每一项是n个，则k = n + 1
    :example:
    该函数以{0}、 {1}、 {2}作为输入，会生成{0,1}、 {0,2}以及{1,2}
    """
    retList = []
    lenLk = len(Ln)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):  # 每一个item都跟其后面的item进行合并，可是如果直接合并是存在问题的
            L1 = list(Ln[i])[: k - 2]  # 这种处理方式真的是太妙了
            L2 = list(Ln[j])[: k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Ln[i] | Ln[j])  # 这里使用集合操作
    return retList


def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)  # 创建项集元素为1的频繁集
    D = list(map(set, dataSet))
    L1, supportData = ScanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2])) > 0:  # 由于K从2开始，所以这里需要减去2，可以debug或者手动执行一下就清楚了
        Ck = aprioriGen(L[k - 2], k)
        Lk, supK = ScanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData


def calcConf(freqSet, H, supportData, bigRuleList, minConf=0.7) ->list:
    """

    :param freqSet:
    :param H:
    :param supportData:
    :param bigRuleList:
    :param minConf:
    :return:
    """
    prundH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '------', conseq, "conf:", conf)
            bigRuleList.append((freqSet - conseq, conseq, conf))
            prundH.append(conseq)
    return prundH


def rulesFromConseq(freqSet, H, supportData, bigRuleList, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > (m + 1):
        Hmpl = aprioriGen(H, m + 1)
        Hmpl = calcConf(freqSet, Hmpl, supportData, bigRuleList, minConf)
        if len(Hmpl) > 1:
            rulesFromConseq(freqSet, Hmpl, supportData, bigRuleList, minConf)


def generateRules(L, supportData, minConf=0.7):
    """
    生成关联规则
    :param L: 频繁集列表
    :param supportData:
    :param minConf: 最小confidence
    :return:
    """
    bigRuleList = []
    for i in range(1, len(L)):  # 从至少两个元素的开始算起，因为一个作为前件，一个作为后件，所以是从1开始
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]  #从频繁集把把每一项取出来
            if i > 1:  # 判断是不是第一个，由于第一个都是两个元素的项，所以可以直接计算，多于两个就需要使用rulesFromConseq
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList
