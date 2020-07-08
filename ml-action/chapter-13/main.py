#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/13 10:45
#@email:chenbinkria@163.com
"""
import pca
import numpy as np

if __name__ == '__main__':
    array = np.array([
        [1, 0.1, 0, 2, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1]
    ])
    matrix = np.mat(array)
    print(matrix*matrix.T)

