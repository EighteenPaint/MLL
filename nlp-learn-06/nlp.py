#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: nlp.py
#@time: 2020/7/3 9:55
#@email:chenbinkria@163.com
"""
# 使用nltk自带的贝叶斯分类器进行性别推断
from nltk.corpus import names
import random

names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in
                                                                 names.words('female.txt')])
import nltk

random.shuffle(names)
feature_sets = [({feature_name: feature_name}, sex) for (feature_name, sex) in names]
train_set = dict(names)
classifier = nltk.NaiveBayesClassifier.train(train_set)  # 这里你可以先提取特征在进行训练，也可以直接训练，把名字本身当成特征,不管文字还是图像，特征提取是一个技术活也是一个体力活

# nltk 的内置决策树分类器，对于已经拥有的分类器，我们要做的是，提供他需要的数据集
# nltk提供了大量开箱即用的NLP工具
