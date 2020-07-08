#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: nlp.py
#@time: 2020/7/1 9:47
#@email:chenbinkria@163.com
"""
import nltk
import matplotlib.pyplot as plt
import numpy as np

nltk.data.path.append(r"E:\dataset\nltk_data")  # 手动修改路径数据，设置NLTK_DATA环境变量无效
from nltk.corpus import brown

news_text = brown.words(categories='news')
print(len(news_text))
fdist = nltk.FreqDist([w.lower() for w in news_text])  # 根据文本进行词频统计
modals = ['can', 'could', 'may', 'might', 'must', 'will']
for m in modals:
    print(m, ":", fdist[m])
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]
plt.bar(list(fdist.keys())[1:50], list(fdist.values())[1:50], alpha=0.5, width=0.3, color='yellow', edgecolor='red',
        label='The First Bar', lw=3)
plt.xlabel("word")
plt.ylabel("freq")
plt.xticks(rotation=90)
plt.show()

# 带条件的频率统计,按照题材分类
cfd = nltk.ConditionalFreqDist(
    [(genre, word) for genre in brown.categories() for word in brown.words(categories=genre)])
genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
modals = ['can', 'could', 'may', 'might', 'must', 'will']
cfd.tabulate(conditions=genres, samples=modals)
"""output
                  can could   may might  must  will 
           news    93    86    66    38    50   389 
       religion    82    59    78    12    54    71 
        hobbies   268    58   131    22    83   264 
science_fiction    16    49     4    12     8    16 
        romance    74   193    11    51    45    43 
          humor    16    30     8     8     9    13 
"""
# 使用plot绘制表格
col_labels = genres

row_labels = modals

table_vals = [cfd[genre][word] for genre in col_labels for word in modals]

row_colors = ['red', 'gold', 'green']

my_table = plt.table(cellText=np.reshape(table_vals, (6, 6)), colWidths=[0.2] * 6, rowLabels=row_labels,
                     colLabels=col_labels, rowColours=row_colors, colColours=row_colors, loc='best')

# 使用双连词生成随机文本,
sent = ['In', 'the', 'beginning', 'God', 'created', 'the', 'heaven', 'and', 'the', 'earth', '.']
nltk.bigrams(sent)  # 这是一个生成器
"""
原理:给定一个词，然后根据以该词开头，后面最有可能出现的词即为下一个单词
"""


def generator_model(cfdist, word, num=15):
    for index in range(num):
        print(word,end='')
        word = cfdist[word].max()  # 输出频数最大的词


text = nltk.corpus.genesis.words("english-kjv.txt")
biggrams = nltk.bigrams(text)  # 会生成双词
cfd = nltk.ConditionalFreqDist(biggrams)  # 就会根据每一双词生成频数
# 问题在于会出现循环文本，比如当我输入
generator_model(cfd,'It')  # 会存在文本循环，单纯通过频率预测会存在很多问题