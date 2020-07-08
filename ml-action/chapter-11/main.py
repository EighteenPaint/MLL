#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/10 11:06
#@email:chenbinkria@163.com
"""
from pprint import pprint

import apriori

if __name__ == '__main__':
    dataSet = apriori.loadDataSet()
    L ,supportData = apriori.apriori(dataSet, minSupport=0.5)
    rules = apriori.generateRules(L, supportData)
    pprint(rules)
