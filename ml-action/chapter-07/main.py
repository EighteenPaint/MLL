#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/6 16:23
#@email:chenbinkria@163.com
"""
import adaboost as ada
if __name__ == '__main__':
    dataArr,classLabel = ada.loadSimpData()
    ada.adaBoostTrainDS(dataArr,classLabel,10)