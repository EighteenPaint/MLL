#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: exercise.py
#@time: 2020/7/1 16:49
#@email:chenbinkria@163.com
"""
import nltk

nltk.data.path.append(r"E:\dataset\nltk_data")
"""
2.用语料库模块处理austen-persuasion.txt。这本书中有多少词标识符？多少词
类型
"""

# 第二题 标识符 = 空格 + 符号 + 单词
from nltk.corpus import gutenberg

res = gutenberg.raw("austen-persuasion.txt")
print("共有标识符：", len(res))
res = gutenberg.words("austen-persuasion.txt")
print("共有词类型：", len(set(res)))
"""
4.用state_union语料库阅读器，访问《国情咨文报告》的文本。计数每个文档中
出现的 men、 women和people。随时间的推移这些词的用法有什么变化？
关键要熟练ConditionalFreqDist以及推导式的使用
"""
# 第四题
from nltk.corpus import state_union

cfd = nltk.ConditionalFreqDist(
    (target, fileid[:4]) for fileid in state_union.fileids for w in state_union.words(fileid) for target in
    ['men', 'women', 'people'] if w.lower().startswith(target))
cfd.plot()

"""
5. ○考查一些名词的整体部分关系。 请记住， 有 3种整体部分关系， 所以你需要使用me
mber_meronyms()， part_meronyms()， substance_meronyms()， member
_holonyms(), part_holonyms()以及substance_holonyms()。
"""
# 第五题
# 此题是wordnet使用，另外还有反义词，蕴含
from nltk.corpus import wordnet as wn

tree = wn.synset('tree.n.01')
tree.member_meronyms()  # 包含tree的部分--部分词
tree.part_meronyms()  # 子结构 --部分词
tree.substance_meronyms()  # 物质-部分词
tree.member_holonyms()  # 包含树的 --上位次，这里一般是森林
tree.part_holonyms()  # 子结构--上位词，这里肯定为空
tree.substance_holonyms()  # 物质--上位词，一般也为空，因为这个集合不想交
"""
6. ○在比较词表的讨论中， 我们创建了一个对象叫做translate， 通过它你可以使用德语
和意大利语词汇查找对应的英语词汇。这种方法可能会出现什么问题？你能提出一个办
法来避免这个问题吗？
如何知道输入的语言是德语还是意大利语呢，特别是在意大利语与德语词汇相同当语义不同的时候
"""
# 第六题
from nltk.corpus import swadesh

fr2en = swadesh.entries(['fr', 'en'])
translate = dict(fr2en)
translate.update(swadesh.entries(['it', 'en']))
# 可能会存在的问题: 如何知道输入的语言是德语还是意大利语呢，特别是在意大利语与德语词汇相同当语义不同的时候

"""
7. ○根据Strunk和 White 的《 Elements of Style》， 词 however在句子开头使用是“ in wh
atever way” 或“ to whatever extent” 的意思，而没有“ nevertheless” 的意思。他们给
出了正确用法的例子： However you advise him, he will probably do as he thinks bes
t.（ http://www.bartleby.com/141/strunk3.html）。使用词汇索引工具在我们一直在思考的
各种文本中研究这个词的实际用法。也可以看 LanguageLog发布在 http://itre.cis.upenn.
edu/~myl/languagelog/archives/001913.html上的 “ Fossilized prejudices about ‘ however’”。
"""

"""
8. ◑在名字语料库上定义一个条件频率分布，显示哪个首字母在男性名字中比在女性名字
中更常用（见图 2-7）。
"""
# 第8题：考察条件频率分布
from nltk.corpus import names

cfd = nltk.ConditionalFreqDist(
    (fileid.split(".")[0], name[0]) for fileid in names.fileids() for name in names.words(fileid))
cfd.plot()
male = cfd['male']
for key in male:
    if male[key] > cfd['female'][key]:
        print(male[key])
"""
9. ◑挑选两个文本， 研究它们之间在词汇、 词汇丰富性、 文体等方面的差异。 你能找出几
个在这两个文本中词意相当不同的词吗？例如： 在《 白鲸记》与《 理智与情感》中的 m
onstrous。
"""
# 丰富性就是词汇量
from nltk.book import *

print(len(text1))  # text1为白鲸记,是text格式,不必写作len(text1.words())
print(len(set(text1)))  # 白鲸记中的词汇量
print(len(set(text2)))  # 理智与情感中的词汇量

# 词意比较
text1.similar('monstrous')
text2.similar('monstrous')

"""
18. ◑写一个程序， 输出一个文本中50个最常见的双连词（ 相邻词对）， 忽略包含停用词的
双连词。
"""
text1.collocations(50, 2)  # 该方法会自己会忽略stopwords
# 如果要输出输出文本中50个最常见的双连词，可以使用条件频率分布
cfd = nltk.ConditionalFreqDist(text1.collocation_list(num=len(text1), window_size=2))
freq_dict = dict(cfd)
sortedFreq = sorted([(first,second) for first in freq_dict for second in freq_dict[first]], key=lambda key: freq_dict[key[0]][key[1]])
out = sortedFreq[:50]

