#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/2 15:46
"""
import logistic
if __name__ == '__main__':
    dataArr,labelMat = logistic.loadDataSet()
    weights = logistic.gradAscent(dataArr, labelMat)
    print(weights)