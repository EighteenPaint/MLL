#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/15 22:33
#@email:chenbinkria@163.com
"""
import svd

if __name__ == '__main__':
    dataMat = svd.loadExData3()
    result = svd.recommand(dataMat, 2)
    print(result)
