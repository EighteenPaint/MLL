#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: name-disambiguation.py
#@time: 2020/8/1 21:35
#@email:chenbinkria@163.com
"""
import random

"""
问题描述：同名但不同作者区分，即同名消歧
"""
# 涉及到的知识点 :
# 读取文本并进行分词
# 并提取特征向量
#   参考：
#   https://www.bilibili.com/video/BV1Vt4y1X799

#
#
# 聚类算法(使用K相似度聚类)
# 基本假设: 论文涉及的领域较为固定
# 思路：
# step1：提取abstract信息，然后生成转化成特征向量
# step2：把同名的论文放在一起作为一个待聚簇数据集，然后进行聚簇，可选用DBSCAN
# 优化思路：
#   1.除了摘要信息，还可以加入共同作者，组织机构，信息越多，分类越精确
# 数据准备：不用了，可以直接收集相关领域的专业词汇，这个应该比较容易，然后使用one hot编码，这样一来特征向量就完成了
# 第二种思路，生成每一个词的词向量，然后求均值作为一个摘要的向量
"""
Penny   演员  表演  剧本 动作
Penny   股票  工作日  收入  
Penny   电影  路演   
Sheldon 量子力学 波函数坍塌 弦理论 降维打击
Sheldon 火星 星球大战 
Sheldon 薛定谔 波粒二象性 
  
"""
# 然后进行聚类
words = list()
cs_file_name = "cs-word"
medicel_file_name = "medical-word"
finance_file_name = "finance-word"
file = [cs_file_name, medicel_file_name, finance_file_name]
sample = dict()
start = 0
# 构建领域语料
for file_name in file:
    abstracts = []  # 样本摘要列表，目前
    with open(file_name) as file:
        for line in file.readlines():
            words.extend(line.split())
    # 每个领域生成10条数据
    end = len(words)  # 获取最后一个元素的
    for data in range(100):  # 生成10份摘要
        abstract_generate_index = random.sample(range(start, end), k=10)  # 每份摘要10个单词
        # print(file_name,abstract_generate_index)
        abstracts.append([words[indz] for indz in abstract_generate_index])
    start = end
    # 保存样本
    sample[file_name] = abstracts
print("语料大小:", len(words))
# print(words)
# print(sample)

from sklearn.feature_extraction.text import CountVectorizer

vecizer = CountVectorizer()
vecizer.fit(words)  # 进行训练
mat = vecizer.transform(
    ["Foreground Background Process Disk Physical Virtual Multithreading Deadlock Page management"]).toarray()  # 得到稀疏矩阵
# print(mat[0])
# 将样本转化为向量
word_vectors = []
for vals in sample.values():
    for val in vals:
        # 转化为长句
        val2String = ' '.join(val)
        input = [val2String]
        vect = vecizer.transform(input).toarray()[0]
        word_vectors.append(vect)
# print(word_vectors[0:5])
#
from sklearn.cluster import DBSCAN
# 半径的选择
from sklearn.metrics.pairwise import euclidean_distances

# distance = euclidean_distances([word_vect for word_vect in word_vectors])

import pandas as pd
import numpy as np

# dis_df = pd.DataFrame(distance)
eps_stat_info = []
# for eps in np.arange(start=3.4,stop=5,step=0.1):
#     print("eps:",eps)
#     for min_sample in np.arange(start=2,stop=20,step=1):
#         print("min_sample:",min_sample)
#         y_pred = DBSCAN(eps=eps,min_samples=min_sample).fit_predict(word_vectors)
#         cluster = set([clazz for clazz in y_pred if clazz != -1])
#         eps_stat_info.append({'eps': eps, 'n_clusters': cluster,"min_sample":min_sample})
# print(pd.DataFrame(eps_stat_info))
y_pred = DBSCAN(eps=3.8, min_samples=5).fit_predict(word_vectors)
print("result:", y_pred)
# 欧式距离是否可以很好的:对于one hot 编码，特别是居于频率的one hot编码的数据，可以很好的定量分析

# 思路2：居于神经网络监督学习的想法，将问题转化为分类问题，我们想要找到一个空间变换，将他们向量映射到领域空间（这是我们定义的），只不过实际上又变成分类问题
# 思路3
# one hot 虽然解决了次序问题，并且可以很好的使用欧式距离，但是矩阵过于稀疏，考虑是否可以使用向量，需要满足
# 1.不受次序影响（对向量求均值，这样），这里的向量可以结合思路2的方式，想办法求个空间变换（参考word2vec，基于神经网络的方式求个映射）
# 2.使用向量余弦相似度进行近邻聚类，需要人工确定
def loadData(path):
    df = pd.read_json(path)
    print(df)
if __name__ == '__main__':
    loadData("/Users/benny.chen/Desktop/datas/name-dis/train/train_pub.json")