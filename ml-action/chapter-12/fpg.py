#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: fpg.py
#@time: 2020/6/11 9:55
#@email:chenbinkria@163.com
"""
import numpy as np
from treeNode import treeNode


def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink is not None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def updateTree(orderedItems, retTree: treeNode, headTable, count):
    if orderedItems[0] in retTree.children:  # 判断FP树中是否已经有了该节点
        retTree.children[orderedItems[0]].inc(count)  # 如果有了就直接增加相应的计数
    else:
        retTree.children[orderedItems[0]] = treeNode(orderedItems[0], count, retTree)  # 如果没有就新增该节点
        if headTable[orderedItems[0]][1] is None:  # 如果头指针指向还没有的话可以指向这个新增的节点
            headTable[orderedItems[0]][1] = retTree.children[orderedItems[0]]
        else:
            updateHeader(headTable[orderedItems[0]][1], retTree.children[orderedItems[0]])  # 如果已经存在就放在已经存在节点的子节点
    if len(orderedItems) > 1:
        updateTree(orderedItems[1::], retTree.children[orderedItems[0]], headTable, count)  # 递归创建


def createTree(dataSet, minSup=1):
    """
    创建 FP 树
    :param dataSet: 训练数据集
        :example
        {
            frozenset({'z'}): 1,
            frozenset({'h', 'j', 'p', 'r', 'z'}): 1,
            frozenset({'t', 'w', 'u', 'v', 'z', 's', 'x', 'y'}): 1,
            frozenset({'n', 'o', 's', 'x', 'r'}): 1,
            frozenset({'t', 'y', 'q', 'p', 'x', 'r', 'z'}): 1,
            frozenset({'t', 'y', 'q', 'm', 'e', 's', 'x', 'z'}): 1
        }
    :param minSup: 最小支持度
    :return:
    """
    headTable = {}  # 头指针表
    """
    创建头指针表
    """
    for trans in dataSet:
        for item in trans:
            headTable[item] = headTable.get(item, 0) + dataSet[trans]
    keys = list(headTable.keys())
    for k in keys:  # 不可以在迭代的时候改变大小
        if headTable[k] < minSup:
            del (headTable[k])
    freqItemSet = set(headTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    for k in headTable:
        headTable[k] = [headTable[k], None]  # 因为value还需要一个指向树节点的指针，故将value扩展为[count,node] 形式
    retTree = treeNode('Null Set', 1, None)
    """
    生成本地数据集，不带树节点
    """
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headTable, count)
    return retTree, headTable


def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


def ascendTree(leafNode, prefixPath):
    if leafNode.parent is not None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(treeNode):
    condPaths = {}
    while treeNode is not None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPaths[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPaths


def mineTree(inTree, headerTable, minSup, prefix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    for basePat in bigL:
        newFreqSet = prefix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead is not None:
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
