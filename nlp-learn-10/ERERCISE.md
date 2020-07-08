1. ○将下列句子翻译成命题逻辑，并用 LogicParser验证结果。提供显示你的翻译中命
题变量如何对应英语表达的一个要点。
a. If Angus sings, it is not the case that Bertie sulks.
b. Cyril runs and barks.
c. It will snow if it doesn’ t rain.
d. It’ s not the case that Irene will be happy if Olive or Tofu comes.
e. Pat didn’ t cough or sneeze.
f. If you don’ t come if I call, I won’ t come if you call.
2. ○翻译下面的句子为一阶逻辑的谓词参数公式。
a. Angus likes Cyril and Irene hates Cyril.
b. Tofu is taller than Bertie.
c. Bruce loves himself and Pat does too.
d. Cyril saw Bertie, but Angus didn’ t.
e. Cyril is a four-legged friend.
f. Tofu and Olive are near each other.
3. ○翻译下列句子为成一阶逻辑的量化公式。
a. Angus likes someone and someone likes Julia.
b. Angus loves a dog who loves him.
c. Nobody smiles at Pat.
d. Somebody coughs and sneezes.
e. Nobody coughed or sneezed.
f. Bruce loves somebody other than Bruce.
g. Nobody other than Matthew loves Pat.
h. Cyril likes everyone except for Irene.
i. Exactly one person is asleep.
4. ○使用λ-抽象和一阶逻辑的量化公式，翻译下列动词短语。
a. feed Cyril and give a capuccino to Angus
b. be given ‘ War and Peace’ by Pat
c. be loved by everyone
d. be loved or detested by everyone
e. be loved by everyone and detested by no-one
5. ○思考下面的语句：
>>> lp = nltk.LogicParser()
>>> e2 = lp.parse('pat')
>>> e3 = nltk.ApplicationExpression(e1, e2)
>>> print e3.simplify()
exists y.love(pat, y)
显然这里缺少了什么东西，即e1值的声明。为了 ApplicationExpression(e1, e2)
被β-转换为exists y.love(pat, y)， e1必须是一个以pat为参数的λ-抽象。 你的任务是
构建这样的一个抽象，将它绑定到 e1，使上面的语句都是满足（上到字母方差）。此外，提
供一个e3.simplify()的非正式的英文翻译。
现在根据e3.simplify()的进一步情况（如下所示）继续做同样的任务：
>>> print e3.simplify()
exists y.(love(pat,y) | love(y,pat))
>>> print e3.simplify()
exists y.(love(pat,y) | love(y,pat))
>>> print e3.simplify()
walk(fido)
6. ○如前面的练习中那样，找到一个λ-抽象e1，产生与下面显示的等效的结果：
>>> e2 = lp.parse('chase')
>>> e3 = nltk.ApplicationExpression(e1, e2)
>>> print e3.simplify()
\x.all y.(dog(y) -> chase(x,pat))
>>> e2 = lp.parse('chase')
>>> e3 = nltk.ApplicationExpression(e1, e2)
>>> print e3.simplify()
\x.exists y.(dog(y) & chase(pat,x))
>>> e2 = lp.parse('give')
>>> e3 = nltk.ApplicationExpression(e1, e2)
>>> print e3.simplify()
\x0 x1.exists y.(present(y) & give(x1,y,x0))
7. ○如前面的练习中那样，找到一个λ-抽象e1，产生与下面显示的等效的结果：348
>>> e2 = lp.parse('bark')
>>> e3 = nltk.ApplicationExpression(e1, e2)
>>> print e3.simplify()
exists y.(dog(x) & bark(x))
>>> e2 = lp.parse('bark')
>>> e3 = nltk.ApplicationExpression(e1, e2)
>>> print e3.simplify()
bark(fido)
>>> e2 = lp.parse('\\P. all x. (dog(x) -> P(x))')
>>> e3 = nltk.ApplicationExpression(e1, e2)
>>> print e3.simplify()
all x.(dog(x) -> bark(x))
8. ◑开发一种方法， 翻译英语句子为带有二元广义量词的公式。 在此方法中， 给定广义量
词Q，量化公式的形式为Q(A, B)，其中A和B是<e, t>类型的表达式。那么，例
如： all(A,B)为真当且仅当A表示的是B所表示的一个子集。
9. ◑扩展前面练习中的方法，使量词如 most和exactly three的真值条件可以在模型中计
算。
10. ◑修改sem.evaluate代码，使它能提供一个有用的错误消息，如果一个表达式不在
模型的估值函数的域中。
11. ●从儿童读物中选择三个或四个连续的句子。一个例子是nltk.corpus.gutenberg:
bryantstories.txt，burgess-busterbrown.txt和edgeworth-parents.txt中的故
事集。 开发一个文法， 能将你的句子翻译成一阶逻辑， 建立一个模型， 使它能检查这些
翻译为真或为假。
12. ●实施前面的练习，但使用DRT作为意思表示。
13. ●以(Warren & Pereira, 1982)为出发点， 开发一种技术， 转换一个自然语言查询为一种
可以更加有效的在模型中评估的形式。例如：给定一个(P(x) & Q(x))形式的查询，将
它转换为(Q(x) & P(x))，如果Q的范围比P小