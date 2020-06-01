#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: main.py
#@time: 2020/6/1 20:49
"""
import bayes
import pprint
if __name__ == '__main__':
    listPosts,listClasses = bayes.loadDataSet()
    myVocabList = bayes.createVocabList(listPosts)
    trainMat = []
    for postDoc in listPosts:
        trainMat.append(bayes.setOfWord2Vec(myVocabList,postDoc))
    p0v, p1v, pab = bayes.trainNB0(trainMat,listClasses)
    print("P(C=侮辱性文档):",pab)
    print(p1v)