#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/5 11:50
#@email:chenbinkria@163.com
"""
import svm

if __name__ == '__main__':
    dataArr,labelArr = svm.loadDataSet("testSet.txt")
    b,alphas = svm.smoSimple(dataArr,labelArr,0.6,0.001,40)