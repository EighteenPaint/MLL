#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: nlp.py
#@time: 2020/7/2 18:01
#@email:chenbinkria@163.com
"""
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

# 默认标注器
import nltk

default_tagger = nltk.DefaultTagger('NN')  # 将NN设置为默认标注
example_words = ['the', 'expensive', 'shop', 'is', 'opening', 'in', 'arcade', 'of', 'New York']
default_tagger.tag(example_words)

# 使用正则表达式方式进行
"""
对于ed结尾，ing结尾,s结尾的单词，可以形成一种规律，当然这种规律依然无法应对所有的单词，也会造成大量的错误
"""
# 查询标注器：使用已经标注的文本作为信息去标注未标注的文本
fd = nltk.FreqDist(brown.words(categories='news'))
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words = list(fd.keys())[:100]
likely_tags = dict((word, cfd[word].max()) for word in most_freq_words)  # max表示只选用最有可能的标注
baseline_tagger = nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)
baseline_tagger.tag(example_words)  # 进行标记:[('the', 'AT'), ('expensive', None), ('shop', None), ('is', None),
# ('opening', None), ('in', 'IN'), ('arcade', None), ('of', 'IN'), ('New York', None)]

# N元标注器
bigram_tagger = nltk.BigramTagger(brown_tagged_sents)  # 我大概知道意思了：记录一个模式：前面N-1个词的词性+当前词的词性，在进行标记的时候如果，模式匹配成功，则就进行推断
test_str = ['The', 'Fulton', 'County', 'Fulton']  # 如果没有理解错的，第一个Fulton会被标记为NP-TL，后面一个有可能标记为None
bigram_tagger.tag(test_str)
"""output
[('The', 'AT'), ('Fulton', 'NP-TL'), ('County', 'NN-TL'), ('Fulton', None)] 与预期结果一致
"""

# 组合标注器
"""
1. 尝试使用 bigram标注器标注标识符。
2. 如果 bigram标注器无法找到一个标记，尝试unigram 标注器。
3. 如果 unigram标注器也无法找到一个标记，使用默认标注器。
"""
t0 = nltk.DefaultTagger('NN')
t1 = nltk.UnigramTagger(brown_tagged_sents, backoff=t0)  # 使用back off参数来进行回退，当无法标记时进行回退，使用回退标记器所指定的标记器进行标记,cutoff,忽略最前面的几个词
t2 = nltk.BigramTagger(brown_tagged_sents, backoff=t1)
t2.evaluate(test_str)

# 标注生词
"""
其实在前面模式匹配的情况下，后面选择最大可能的类型，这个貌似没有直接的API，需要自己写

"""
# Brill标注器：不断的细化，这个应该可以用来标注生词
# 这个应该会比较强大，但是书中讲的比较少，不知道后面会不会涉及到

