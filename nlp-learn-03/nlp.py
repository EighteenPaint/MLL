#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: nlp.py
#@time: 2020/7/2 10:23
#@email:chenbinkria@163.com
"""
# 词干提取器:将一些单词的变形转换为原来样子，不如现在分词和过去分词,复数
import nltk
nltk.data.path.append(r"E:\dataset\nltk_data")
nltk.data.root
porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
porter.stem('showed')  # 会转化为show
porter.stem("lying")  # 会转化为lie
# word net提供一个可以处理其他变形的单词
wnl = nltk.WordNetLemmatizer()
wnl.lemmatize('women')
