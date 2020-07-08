#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/8 21:42
#@email:chenbinkria@163.com
"""
import regTree as rt
import numpy as np

if __name__ == '__main__':
    myData1 = rt.loadDataSet("ex0.txt")
    mymat1 = np.mat(myData1)
    regTree = rt.createTree(mymat1)
    print(regTree)