#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: breast-cancer.py
#@time: 2020/6/18 21:59
#@email:chenbinkria@163.com
"""
from sklearn.datasets import load_breast_cancer
import dtree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cancers = load_breast_cancer()
X = cancers.data  # 获取特征值
Y = cancers.target  # 获取标签

feature_names = cancers.feature_names
df = pd.DataFrame(X, columns=feature_names)
df_label = pd.DataFrame(Y, columns=["target"])

res = pd.concat([df, df_label], axis=1, ignore_index=False)
dataset = np.array(res).astype(float)
labels = list(feature_names)
#todo:起始时间
tree = dtree.createTree(dataset, labels)
#todo:结束时间
print(tree)
