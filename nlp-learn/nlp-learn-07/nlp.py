#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: nlp.py
#@time: 2020/7/4 16:47
#@email:chenbinkria@163.com
"""
"""
1. IOB格式分类标注标识符为I、 O和 B。三个标签为什么是必要的？如果我们只使用
I和 O标记会造成什么问题？
答:Inside,Outside,Begin:如果没有Begin，则没有办法区分边界，一个块的开始到下一个块的开始才会被认为是一个块
"""
# 答:Inside,Outside,Begin:如果没有Begin，则没有办法区分边界，一个块的开始到下一个块的开始才会被认为是一个块
"""
2. 写一个标记模式匹配包含复数中心名词在内的名词短语， 如many/JJ researchers
/NNS, two/CD weeks/NNS, both/DT new/JJ positions/NNS。通过泛化处理单
数名词短语的标记模式，尝试做这个。
"""
# NP:名词短语
# VP:动词短语
# PP:
from nltk.corpus import conll2000

conll2000.chunked_sents()
# nltk的正则表达式,有单独含义，可以通过词性进行分块，所以前提就是词性标记已经完成，词性标记是前面章节的内容
# NN 常用名词 单数形式
#
# NNS 常用名词 复数形式
#
# NNP 专有名词 单数形式
#
# NNPS 专有名词 复数形式

grammer = r"NP:{<JJ|CD|DT><JJ>?<NNS>}"  # 开头是JJ,或者CD，或者DT,注意<JJ|CD|DT><JJ>的写法，<>是nltk的不是正则表达式的
import nltk

rp = nltk.RegexpParser(grammer)
sentens = ["many/JJ", "researchers/NNS", "two/CD", "weeks/NNS", "both/DT", "new/JJ", "positions/NNS"]
sents = [nltk.str2tuple(str) for str in sentens]
rp.parse(sents)
"""
3. 选择CoNLL-2000分块语料库中三种块类型之一。查看这些数据， 并尝试观察组成这
种类型的块的 POS标记序列的任一模式。 使用正则表达式分块器nltk.RegexpParser
开发一个简单的分块器。讨论任何难以可靠分块的标记序列。
"""
# 查看三种语料库之一，我们选择NP类型
from nltk.corpus import conll2000

tree = conll2000.chunked_sents(chunk_types=['NP'])
tree.draw()
"""
4. 块的早期定义是出现在缝隙之间的材料。开发一个分块器以将完整的句子作为一个单
独的块开始， 然后其余的工作完全由加缝隙完成。 在你自己的应用程序的帮助下， 确定
哪些标记（ 或标记序列）最有可能组成缝隙。 相对于完全基于块规则的分块器比较这种
方法的性能和易用性。
"""
# 间隙定义有自己模式：}<tag>{
"""
7. 用任何你之前已经开发的分块器执行下列评估任务。（请注意，大多数分块语料库包
含一些内部的不一致，以至于任何合理的基于规则的方法都将产生错误。）
a. 在来自分块语料库的 100个句子上评估你的分块器， 报告精度、 召回率和F量度。
b. 使用 chunkscore.missed()和 chunkscore.incorrect()方法识别你的分块器的
错误，并讨论它。
"""
# 分块器评估的学习,以及chunksore的使用
from nltk.corpus import conll2000

cp = nltk.RegexpParser("")
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
data = conll2000.tagged_words('test.txt')
print(cp.evaluate(test_sents))
grammer = r"NP: {<[CDJNP].*>+}"
cp = nltk.RegexpParser(grammer)
chunkscore = nltk.chunk.ChunkScore()
guess = cp.parse(data)
chunkscore.score(test_sents, guess)
chunkscore.missed()
chunkscore.incorrect()
"""
10. bigram分块器的准确性得分约为 90％。研究它的错误，并试图找出它为什么不能获
得 100％的准确率。实验trigram分块。你能够在提高性能吗？
"""


# 考察第二种分块器：词袋分块器
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t, c) for w, t, c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]  # 获取每一个单词在块中的标记，作为训练集，也就是说，对于分块来说，也是一种标记操作,什么意思呢？
        """
        train_data = [[('NNP', 'B-NP'), ('NNP', 'I-NP'), ('NNP', 'I-NP'), ('POS', 'B-NP'), ('NNP', 'I-NP'), ('NN', 'I-NP'), ('VBD', 'O'),
                ('POS', 'B-NP'), ('NN', 'I-NP'), (',', 'O'), ('VBG', 'O'), ('DT', 'B-NP'), ('NN', 'I-NP'), ('VBN', 'O'), ('IN', 'O'), ('DT', 'B-NP'),
                 ('NN', 'I-NP'), ('IN', 'O'), ('NNP', 'B-NP'), ('NNP', 'I-NP'), ('JJ', 'B-NP'), ('NNP', 'I-NP'), ('.', 'O')], [('NNP', 'B-NP'),
                  ('NNP', 'I-NP'), (',', 'O'), ('CD', 'B-NP'), ('NNS', 'I-NP'), ('JJ', 'O'), (',', 'O'), ('VBN', 'O'), ('IN', 'O'), ('NN', 'B-NP'),
                   ('NN', 'I-NP'), ('IN', 'O'), ('DT', 'B-NP'), ('NNP', 'I-NP'), ('NN', 'I-NP'), ('.', 'O')], [('IN', 'O'), ('NNP', 'B-NP'), (',', 'O'), ('PRP', 'B-NP'), ('VBD', 'O'), ('DT', 'B-NP'), ('NN', 'I-NP'), ('IN', 'O'), ('NN', 'B-NP'), ('NN', 'I-NP'), ('IN', 'O'), ('NNP', 'B-NP'), ('NNP', 'I-NP'), (',', 'O'), ('DT', 'B-NP'), ('NN', 'I-NP'), ('NN', 'I-NP'), ('NN', 'I-NP'), ('.', 'O')], [('NNP', 'B-NP'), ('NNP', 'I-NP'), ('NNP', 'I-NP'), ('NNP', 'I-NP'), ('.', 'O')], [('NNP', 'B-NP'), ('NNP', 'I-NP'), ('NNP', 'I-NP'), (',', 'O'), ('CD', 'B-NP'), ('NNS', 'I-NP'), ('JJ', 'O'), (',', 'O'), ('VBD', 'O'), ('VBN', 'O'), ('NN', 'B-NP'), ('CC', 'O'), ('JJ', 'B-NP'), ('VBG', 'I-NP'), ('NN', 'I-NP'), ('IN', 'O'), ('NNP', 'B-NP'), ('NNP', 'I-NP'), (',', 'O'), ('DT', 'B-NP'), ('NNP', 'I-NP'), (',', 'I-NP'), ('NNP', 'I-NP'), (',', 'I-NP'), ('NN', 'I-NP'), ('IN', 'O'), ('DT', 'B-NP'), ('NNP', 'I-NP'), ('NNP', 'I-NP'), ('NN', 'I-NP'), ('NN', 'I-NP'), ('NN', 'I-NP'), ('.', 'O')], [('NNP', 'B-NP'), (',', 'O'), ('WDT', 'B-NP'), ('VBZ', 'O'), ('NNS', 'B-NP'), ('IN', 'O'), ('JJ', 'B-NP'), ('NN', 'I-NP'), (',', 'O'), ('VBD', 'O'), ('DT', 'B-NP'), ('NN', 'I-NP'), ('VBZ', 'O'), ('RB', 'O'), ('VBN', 'O'), ('.', 'O')], [('NNP', 'B-NP'), ('NNP', 'I-NP'), ('VBD', 'O'), ('VBN', 'O'), ('JJ', 'B-NP'), ('NN', 'I-NP'), ('NN', 'I-NP'), ('IN', 'O'), ('NNP', 'B-NP'), ('.', 'O')], [('IN', 'O'), ('NN', 'B-NP'), ('TO', 'O'), ('PRP$', 'B-NP'), ('JJ', 'I-NP'), ('NN', 'I-NP'), ('NN', 'I-NP'), ('CC', 'I-NP'), ('NN', 'I-NP'), ('NNS', 'I-NP'), (',', 'O'), ('NNP', 'B-NP'), ('NNP', 'I-NP'), ('VBZ', 'O'), ('NN', 'B-NP'), ('IN', 'O'), ('NN', 'B-NP'), ('CC', 'O'), ('NN', 'B-NP'), ('NN', 'I-NP'), ('.', 'O')], [('DT', 'B-NP'), ('NNS', 'I-NP'), ('VBD', 'O'), ('VBN', 'O'), ('VBN', 'O'), ('IN', 'O'), ('NNP', 'B-NP'), ('NNP', 'I-NP'), (',', 'O'), ('CD', 'B-NP'), (',', 'O'), ('WP', 'B-NP'), ('VBD', 'O'), ('IN', 'O'), ('DT', 'B-NP'), ('JJ', 'I-NP'), ('NN', 'I-NP'), ('NN', 'I-NP'), ('.', 'O')], [('NNP', 'B-NP'), ('VBZ', 'O'), ('IN', 'O'), ('JJ', 'O'), ('IN', 'O'), ('NNP', 'B-NP'), ('NNP', 'I-NP'), ('NNP', 'I-NP'), ('.', 'O')], [('NNP', 'B-NP'), ('NNP', 'I-NP'), ('NNP', 'I-NP'), (',', 'O'), ('VBG', 'O'), ('JJ', 'B-NP'), ('NNP', 'I-NP'), ('JJ', 'I-NP'), ('NN', 'I-NP'), ('NNS', 'I-NP'), (',', 'O'), ('VBD', 'O'), ('DT', 'B-NP'), ('JJ', 'I-NP'), ('NN', 'I-NP'), ('IN', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('CD', 'I-NP'), (',', 'O'), ('CC', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('DT', 'B-NP'), ('NN', 'I-NP'), ('.', 'O')], [('DT', 'B-NP'), ('NN', 'I-NP'), ('RBR', 'O'), (',', 'O'), ('DT', 'B-NP'), ('NNS', 'I-NP'), ('VBP', 'I-NP'), ('VBD', 'O'), ('NNS', 'B-NP'), ('IN', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('CD', 'I-NP'), (',', 'O'), ('CC', 'O'), ('CD', 'B-NP'), ('NNS', 'I-NP'), ('DT', 'B-NP'), ('NN', 'I-NP'), ('.', 'O')], [('IN', 'O'), ('DT', 'B-NP'), ('CD', 'I-NP'), ('NNS', 'I-NP'), (',', 'O'), ('PRP', 'B-NP'), ('VBD', 'O'), ('DT', 'B-NP'), ('NN', 'I-NP'), ('IN', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('CD', 'I-NP'), (',', 'O'), ('CC', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('DT', 'B-NP'), ('NN', 'I-NP'), (',', 'O'), ('IN', 'O'), ('NNS', 'B-NP'), ('IN', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('CD', 'I-NP'), (',', 'O'), ('CC', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('DT', 'B-NP'), ('NN', 'I-NP'), (',', 'O'), ('IN', 'O'), ('DT', 'B-NP'), ('CD', 'I-NP'), ('NN', 'I-NP'), ('.', 'O')], [('NNP', 'B-NP'), ('NNP', 'I-NP'), ('VBD', 'O'), ('PRP', 'B-NP'), ('VBD', 'O'), ('PRP$', 'B-NP'), ('NN', 'I-NP'), ('NNS', 'I-NP'), ('IN', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('CD', 'I-NP'), ('IN', 'O'), ('VBG', 'O'), ('PRP$', 'B-NP'), ('NN', 'I-NP'), ('NN', 'I-NP'), (',', 'O'), ('VBG', 'O'), ('PRP$', 'B-NP'), ('JJ', 'I-NP'), ('NN', 'I-NP'), ('CC', 'I-NP'), ('JJ', 'I-NP'), ('NN', 'I-NP'), ('NNS', 'I-NP'), ('TO', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('CD', 'I-NP'), ('.', 'O')], [('IN', 'O'), ('DT', 'B-NP'), ('NN', 'I-NP'), ('NN', 'I-NP'), (',', 'O'), ('PRP', 'B-NP'), ('VBD', 'O'), (',', 'O'), ('PRP', 'B-NP'), ('VBD', 'O'), ('VBG', 'B-NP'), 
        ('NN', 'I-NP'), ('IN', 'O'), ('$', 'B-NP'), ('CD', 'I-NP'), ('CD', 'I-NP'), ('IN', 'O'), ('DT', 'B-NP'), ('NN', 'I-NP'), ('.', 'O')]]
        使用UnigramTagger训练，对于NNP我就进行标记为B-NP，这里你会发现，对于同一个NNP会有不同的标记，这也就是问题所在
        """
        self.tagger = nltk.UnigramTagger(train_data)


    def parse(self, sentence):
        pos_tags = [pos for (word, pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word, pos), chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

# 实体标签、实体识别，实体其实在语料库中也是已经标识了