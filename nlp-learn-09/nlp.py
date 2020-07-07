#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: nlp.py
#@time: 2020/7/7 12:01
#@email:chenbinkria@163.com
"""
"""
1. 需要什么样的限制才能正确分析词序列，如I am happy和 she is happy而不是*you
is happy或*they am happy？实现英语中动词be的现在时态范例的两个解决方案，首
先以文法（ 8）作为起点，然后以文法（ 20）为起点。
"""
# 我们可以在文法定义上进行限制,增加单复数属性/特征
# YOU -> 'you' | 'You' BE[NUM=pl]
# BE[NUM=pl] -> 'were' | 'are'    关键看你如何定义文法，并增加特征，其实这样已经非常灵活的，至少生成一个语法正确的语句是没有问题的
# 对于计算机来说，他肯定不会思考，但是归根结底是一个IO系统，我们只需要找到映射即可，就像神经网络，通过训练的方式找到最好的映射
# 一种映射系统或者一种链式推导系统
# Input 我想吃饭：映射为{“我饿了”，“到饭点了”，“我需要”}
# S -> NP[NUM=?n] VP[NUM=?n] ADJ
# ADJ -> "happy"
# NP[NUM=I] -> "I"
# NP[NUM=YOU] -> "You" | "you"
# NP[NUM=H] -> "she" | "he" |"He"|"She"
# VP[NUM=I] -> "am"
# VP[NUM=YOU] -> "are" | "were"
# VP[NUM=H] -> "is" | "was"
import nltk
# grammer = nltk.CFG.fromstring("""
# S -> NP[NUM=?n] VP[NUM=?n] ADJ
# ADJ -> "happy"
# NP[NUM=I] -> "I"
# NP[NUM=YOU] -> "You" | "you"
# NP[NUM=H] -> "she" | "he" |"He"|"She"
# VP[NUM=I] -> "am"
# VP[NUM=YOU] -> "are" | "were"
# VP[NUM=H] -> "is" | "was"
# """)
parser = nltk.load_parser(grammar_url="grammer.fcfg",trace=5)
sent = "she is happy".split()  # 当其错误的时候会无法进行解析
tree = parser.parse(sent)
for tr in tree:
    print(tr)
# 标签是可以嵌套使用的，这样一来就提供了更加灵活的文法定义
