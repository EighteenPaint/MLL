#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: nlp.py
#@time: 2020/7/5 15:52
#@email:chenbinkria@163.com
"""
import  nltk
grammar1 = nltk.grammar.CFG.fromstring("""
S -> NP VP
VP -> V NP | V NP PP
PP -> P NP
V -> "saw" | "ate" | "walked"
NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
Det -> "a" | "an" | "the" | "my"
N -> "man" | "dog" | "cat" | "telescope" | "park"
P -> "in" | "on" | "by" | "with"
""")
sent = "Mary saw Bob".split()
rd_parse = nltk.RecursiveDescentParser(grammar1,trace=0)
for tree in rd_parse.parse(sent):
    print(tree)
"""
3. 思考句子： Kim arrived or Dana left and everyone cheered。用括号的形式表示 and
和 or的相对范围。产生这两种解释对应的树结构。
"""
#(Kim arrived) or ((Dana left) and (everyone cheered))
#(Kim arrived) or (Dana left) and (everyone cheered)
grammer1="""
S -> N VP | "Kim_arrived"
N -> 'Kim' | 'and'
S -> N VP | "Dana" "left" | "Dana" "left" "and" "everyone" "cheered"
CC -> 'or' |'and'
S -> N VP | "everyone" "cheered"
"""
sent = "Kim arrived or Dana left and everyone cheered".split()
gramr = nltk.grammar.CFG.fromstring("""
S ->  SS CC SS CC SS | SS CC SSS  # 三个短句组成或者一个短句和一个并列短句组成
CC -> 'or' | 'and' #CC 是or 或者 and
NP -> PRP | "Kim" | "Dana" #名词短语集合
PRP -> "everyone" #代词集合
VP -> "arrived" | "cheered" | "left" #动词过去分词集合
SS -> NP VP #短句的形式
SSS -> SS CC SS ##并列短句形式
""")
parse = nltk.RecursiveDescentParser(gramr,trace=5)
for tree in parse.parse(sent):
    print(tree)
    tree.draw()
"""
5. ○在本练习中，你将手动构造一些分析树。
a. 编写代码产生两棵树，对应短语 old men and women的不同读法。
b. 将本章中表示的任一一颗树编码为加标签的括号括起的形式， 使用nltk.Tree()检
查它是否符合语法。使用 draw()显示树。
(S
  (ADJ old)
  (NP man and women)  
)
(S
   (NP old man)
   (CC and)
   (NP woman)
)
"""
"""
7. ○分析 A.A. Milne关于 Piglet的句子，为它包含的所有句子画下划线，然后用 S替换
这些（ 如： 第一句话变为S when S）。 为这种“ 压缩”的句子画一个树形结构。 用于建
立这样一个长句的主要的句法结构是什么？
"""
