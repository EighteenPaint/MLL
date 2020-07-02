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
10. ◑阅读BBC 新闻文章：“ UK’ s Vicky Pollards ‘ left behind’” http://news.bbc.co.uk/1/
hi/education/6173441.stm。文章给出了有关青少年语言的以下统计：“ 使用最多的 20个
词，包括 yeah, no, but和like，占所有词的大约三分之一”。对于大量文本源来说，所
有词标识符的三分之一有多少词类型？你从这个统计中得出什么结论？更多相关信息
请阅读LanguageLog上的http://itre.cis.upenn.edu/~myl/languagelog/archives/003993.html。

11. ◑调查模式分布表，寻找其他模式。试着用你自己对不同文体的印象理解来解释它们。
你能找到其他封闭的词汇归类，展现不同文体的显著差异吗？
12. ◑CMU发音词典包含某些词的多个发音。它包含多少种不同的词？具有多个可能的发
音的词在这个词典中的比例是多少？
13. ◑没有下位词的名词同义词集所占的百分比是多少？你可以使用wn.all_synsets('n')
得到所有名词同义词集。
14. ◑定义函数supergloss(s)， 使用一个同义词集 s作为它的参数，返回一个字符串，包
含 s的定义和s所有的上位词与下位词的定义的连接字符串。
"""
# 14
"""
15. ◑写一个程序，找出所有在布朗语料库中出现至少3次的词。
16. ◑写一个程序，生成如表1-1所示的词汇多样性得分表（例如：标识符/类型的比例）。
包括布朗语料库文体的全集 （ nltk.corpus.brown.categories()）。哪个文体词汇多样
性最低（每个类型的标识符数最多）？这是你所期望的吗？
17. ◑写一个函数，找出一个文本中最常出现的50个词，停用词除外。
18. ◑写一个程序， 输出一个文本中50个最常见的双连词（ 相邻词对）， 忽略包含停用词的
双连词。
"""
text1.collocations(50, 2)  # 该方法会自己会忽略stopwords
# 如果要输出输出文本中50个最常见的双连词，可以使用条件频率分布
nltk.ConditionalFreqDist(text1.collocation_list())
"""
19. ◑写一个程序， 按文体创建一个词频表， 以 2.1节给出的词频表为范例， 选择你自己的
词汇，并尝试找出那些在一个文体中很突出或很缺乏的词汇。讨论你的研究结果。
20. ◑写一个函数word_freq()，用一个词和布朗语料库中的一个部分的名字作为参数，
计算这部分语料中词的频率。
21. ◑写一个程序，估算一个文本中的音节数，利用CMU 发音词典。
22. ◑定义一个函数hedge(text)，处理一个文本和产生一个新的版本在每三个词之间插
入一个词 like。87
23. ●齐夫定律： f(w)是一个自由文本中的词 w的频率。假设一个文本中的所有词都按照它
们的频率排名， 频率最高的在最前面。齐夫定律指出一个词类型的频率与它的排名成反
比（ 即 f×r=k， k是某个常数）。 例如： 最常见的第50个词类型出现的频率应该是最常
见的第 150个词型出现频率的 3倍。
a. 写一个函数来处理一个大文本,使用pylab.plot画出相对于词的排名的词的频率。
你认可齐夫定律吗？（提示：使用对数刻度会有帮助。）所绘的线的极端情况是怎
样的？
b. 随机生成文本，如：使用random.choice("abcdefg ")，注意要包括空格字符。
你需要事先import random。使用字符串连接操作将字符累积成一个很长的字符
串。 然后为这个字符串分词， 产生前面的齐夫图， 比较这两个图。 此时你怎么看齐
夫定律？
24. ●修改例2-1的文本生成程序，进一步完成下列任务：
a. 在一个词链表中存储 n个最相似的词，使用random.choice()从链表中随机选取
一个词。（你将需要事先import random）
b. 选择特定的文体， 如： 布朗语料库中的一部分或者《 创世纪》 翻译或者古腾堡语料
库中的文本或者一个网络文本。 在此语料上训练一个模型， 产生随机文本。 你可能
要实验不同的起始字。文本的可理解性如何？讨论这种方法产生随机文本的长处和
短处。
c. 现在使用两种不同文体训练你的系统， 使用混合文体文本做实验。讨论你的观察结
果。
25. ●定义一个函数find_language()，用一个字符串作为其参数，返回包含这个字符串
作为词汇的语言的列表。 使用《 世界人权宣言》（ udhr） 的语料， 将你的搜索限制在L
atin-1编码的文件中。
26. ●名词上位词层次的分枝因素是什么？也就是说，对于每一个具有下位词——上位词层
次中的子女——的名词同义词集， 它们平均有几个下位词？你可以使用wn.all_synse
ts('n')获得所有名词同义词集。
27. ●一个词的多义性是它所有含义的个数。 利用WordNet， 使用len(wn.synsets('dog',
'n'))我们可以判断名词 dog有 7种含义。 计算WordNet中名词、 动词、 形容词和副词
的平均多义性。
28. ●使用预定义的相似性度量之一给下面的每个词对的相似性打分。按相似性减少的顺序
排名。你的排名与这里给出的顺序有多接近？ (Miller & Charles, 1998)实验得出的顺
序：car-automobile, gem-jewel, journey-voyage, boy-lad, coast-shore, asylum-madhouse,
magician-wizard, midday-noon, furnace-stove, food-fruit, bird-cock, bird-crane, tool-imp
lement, brother-monk, lad-brother, crane-implement, journey-car, monk-oracle, cemeterywoodland, food-rooster, coast-hill, 
forest-graveyard, shore-woodland, monk-slave, coast-f
orest, lad-wizard, chord-smile, glass-magician, rooster-voyag
"""
