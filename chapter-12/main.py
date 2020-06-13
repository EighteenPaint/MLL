#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/11 11:15
#@email:chenbinkria@163.com
"""
from pprint import pprint

import fpg

if __name__ == '__main__':
    simData = fpg.loadSimpDat()
    initSet = fpg.createInitSet(simData)
    tree, headTable = fpg.createTree(initSet, 1)
    tree.disp()
    path = fpg.findPrefixPath(headTable['t'][1])
    pprint(path)
    """
    构建条件FP树
    """
    freqItems = []
    fpg.mineTree(tree, headTable, 3, set([]), freqItems)
    pprint(freqItems)
