1. ○需要什么样的限制才能正确分析词序列，如I am happy和 she is happy而不是*you
is happy或*they am happy？实现英语中动词be的现在时态范例的两个解决方案，首
先以文法（ 8）作为起点，然后以文法（ 20）为起点。
2. ○开发例9-1中文法的变体，使用特种COUNT来区分下面显示的句子：
(56) a. The boy sings.
b. *Boy sings.
(57) a. The boys sing.
b. Boys sing.
(58) a. The water is precious.
b. Water is precious.
3. ○写函数subsumes()判断两个特征结构 fs1和fs2是否fs1包含fs2。
4. ○修改（ 30）中所示的文法纳入特征BAR来处理短语投影。
5. ○修改例9-4中的德语文法，加入9.3节中介绍的子类别的处理。
6. ◑开发一个基于特征的文法，能够正确描述下面的西班牙语名词短语：
(59) un cuadro hermos-o
INDEF.SG.MASC picture beautiful-SG.MASC
‘ a beautiful picture’312
(60) un-os cuadro-s hermos-os
INDEF-PL.MASC picture-PL beautiful-PL.MASC
‘ beautiful pictures’
(61) un-a cortina hermos-a
INDEF-SG.FEM curtain beautiful-SG.FEM
‘ a beautiful curtain’
(62) un-as cortina-s hermos-as
INDEF-PL.FEM curtain beautiful-PL.FEM
‘ beautiful curtains’
7. ◑开发earley_parser的包装程序，只在输入序列分析出错时才输出跟踪。
8. ◑思考例9-5的特征结构。
例9-5. 探索特征结构。
fs1 = nltk.FeatStruct("[A = ?x, B= [C = ?x]]")
fs2 = nltk.FeatStruct("[B = [D = d]]")
fs3 = nltk.FeatStruct("[B = [C = d]]")
fs4 = nltk.FeatStruct("[A = (1)[B = b], C->(1)]")
fs5 = nltk.FeatStruct("[A = (1)[D = ?x], C = [E -> (1), F = ?x] ]")
fs6 = nltk.FeatStruct("[A = [D = d]]")
fs7 = nltk.FeatStruct("[A = [D = d], C = [F = [D = d]]]")
fs8 = nltk.FeatStruct("[A = (1)[D = ?x, G = ?x], C = [B = ?x, E -> (1)] ]")
fs9 = nltk.FeatStruct("[A = [B = b], C = [E = [G = e]]]")
fs10 = nltk.FeatStruct("[A = (1)[B = b], C -> (1)]")
在纸上计算下面的统一的结果是什么。（提示：你可能会发现绘制图结构很有用。）
a. fs1 and fs2
b. fs1 and fs3
c. fs4 and fs5
d. fs5 and fs6
e. fs5 and fs7
f. fs8 and fs9
g. fs8 and fs10
用NLTK检查你的答案。
9. ◑列出两个包含[A=?x, B=?x]的特征结构。
10. ◑忽略结构共享，给出一个统一两个特征结构的非正式算法。
11. ◑扩展例9-4中的德语语法，使它能处理所谓的动词第二顺位结构，如下所示：
(63) Heute sieht der Hund die Katze.
12. ◑同义动词的句法属性看上去略有不同(Levin,1993)。思考下面的动词 loaded、 filled和
dumped的文法模式。你能写文法产生式处理这些数据吗？
(64) a. The farmer loaded the cart with sand
b. The farmer loaded sand into the cart
c. The farmer filled the cart with sand
d. *The farmer filled sand into the cart
e. *The farmer dumped the cart with sand
f. The farmer dumped sand into the cart
13. ●形态范例很少是完全正规的， 矩阵中的每个单元的意义有不同的实现。例如： 词位 w313
alk的现在时态词性变化只有两种不同形式：第三人称单数的walks和所有其他人称和
数量的组合的 walk。一个成功的分析不应该额外要求 6个可能的形态组合中有 5个有
相同的实现。设计和实施一个方法处理这个问题。
14. ●所谓的核心特征在父节点和核心孩子节点之间共享。例如： TENSE是核心特征， 在
一个 VP和它的核心孩子 V之间共享。更多细节见(Gazdar et al., 1985)。我们看到的结
构中大部分是核心结构——除了 SUBCAT和SLASH。由于核心特征的共享是可以预
见的， 它不需要在文法产生式中明确表示。开发一种方法自动计算核心结构的这种规则
行为的比重。
15. ●扩展NLTK中特征结构的处理， 允许统一值为链表的特征， 使用这个来实现一个 HP
SG风格的子类别分析，核心类别的SUBCAT是它的补语的类别和它直接父母的 SUB
CAT值的连结。
16. ●扩展NLTK的特征结构处理， 允许带未指定类别的产生式， 例如： S[-INV] -> ?x
S/?x。
17. ●扩展NLTK的特征结构处理，允许指定类型的特征结构。
18. ●挑选一些(Huddleston & Pullum,2002)中描述的文法结构，建立一个基于特征的文法
计算它们的比