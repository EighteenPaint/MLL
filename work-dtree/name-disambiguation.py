#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: name-disambiguation.py
#@time: 2020/8/1 21:35
#@email:chenbinkria@163.com
"""
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
"""
Penny   演员  表演  剧本 动作
Penny   股票  工作日  收入  
Penny   电影  路演   
Sheldon 量子力学 波函数坍塌 弦理论 降维打击
Sheldon 火星 星球大战 
Sheldon 薛定谔 波粒二象性 
  
"""
# 然后进行聚类
