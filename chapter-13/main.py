#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/13 10:45
#@email:chenbinkria@163.com
"""
import pca

if __name__ == '__main__':
    dataMat = pca.loadDataSet('testSet.txt')
    lowDMat, reconMat = pca.pca(dataMat, 1)
    print("降维后的矩阵", lowDMat)
    print("原始数据集：", dataMat)
    print("复原的数据：", reconMat)
