#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: breast-cancer.py
#@time: 2020/6/18 21:59
#@email:chenbinkria@163.com
"""
from sklearn.datasets import load_breast_cancer
import regTrees
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
数据集导入
"""
cancers = load_breast_cancer()
X = cancers.data  # 获取特征值
Y = cancers.target  # 获取标签
feature_names = cancers.feature_names
df = pd.DataFrame(X, columns=feature_names)
df_label = pd.DataFrame(Y, columns=["target"])
res = pd.concat([df, df_label], axis=1, ignore_index=False)
dataset = np.array(res).astype(float)
labels = list(feature_names)
# tree = dtree.createTree(dataset, labels)
# print(tree)
"""
数据集划分
"""
from sklearn.model_selection import KFold

k = 5
kf = KFold(n_splits=k)  # 使用KFold进行数据集划分
myscores = []
for train, test in kf.split(dataset.copy()):
    """
    进行训练
    """
    datasetMat = np.mat(dataset[train])
    tree = regTrees.createTree(datasetMat.copy(), ops=(1, 1))
    finalColumnIndex = np.shape(dataset)[1] - 2
    testDataSet = dataset[test, :-1]
    real_value = dataset[test, -1]
    score = regTrees.preBreast(tree, testDataSet, real_value.astype(int))
    myscores.append(score)
    """
    最后一列是分类，作为预测数据不需要
    """
    # print("tree", tree)
"""
交叉验证
"""
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
scores = cross_val_score(dtc, X, Y, cv=5, scoring='accuracy')
print("SK DT scores: ", scores, "SK正确率均值：", np.mean(scores))
print("MY DT scores: ", myscores, "自实现DT正确率均值", np.mean(myscores))
