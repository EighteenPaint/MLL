# 深度学习用于文本和序列
## 思考与总结
1. 对于深度学习，似乎最关键在于数据能否变成张量，似乎变成了张量，神经网络就无所不能
2. 对于那些有多种解释的词向量，神经网络会自动学习到么，神经网络是基于错误学习的，你只要告诉他，你错了，他自己就会纠正，直到把题作对。
在我看来，只要向量意义符合数学意义，比如几何向量距离，逻辑推理，符合人类认知，不必太担心神经网络能不能真正理解我们想告诉他的含义，而是让其排除掉所有错误答案，那么留下来的就是他需要理解的
3. 如何理解嵌入层是一个字典，其实很好理解，嵌入层 key:value,key就是单词的整数索引,value就是对应的词向量，Embedding(word) = vector
4. 再次说一遍神经网络的核心：找到一个最好的新空间的表示
5. 编码:只要支持编码与原信息的相互转化都是合理的编码，不过一般来说，我们都会选择尽量短小而且抗干扰性强的编码方式
6. 词嵌入就是用一个指定长度的向量来表示一个单词，甚至在初始的时候，这个向量可以是任意的，但只要可以唯一的指代这个单词即可。但是向量间的几何关系应该能够反映某种关系
7. 比如一个单词具有特征：词性，情感，类型，语义等，在传统机器学习我们需要时间去做这些工作，但是在神经网络时代，我们不需要，神经网络会通过不断学习找到一个合适的向量来包含我们想到甚至没想到的特征，只要其维度足够大
从一个单词到一个向量就是一种空间转换，向量空间（甚至说张量空间）就是我们所说的新空间，只是越复杂的对象，其表示越复杂。
8. 只要给与神经网络足够的时间和数据，他总能找到新空间，并同时找到相对好的映射
9. 神经网络把所有的数据都采用某种方式变成向量，这个向量会包含所有有用特征，甚至代表数据本身，越多的特征往往意味着越多的维度。其实就是前面所说的f（g（a(b(c....))
10. 与训练的词向量:已经有前辈将各个单词训练好成了向量而且其特征也相对较多
11. 看是否过拟合，就看训练精度与检验精度是否呈反方向变化
12. 文本训练容易过拟合在于，训练集太少（经验：数据集越少，训练时间越长，过拟合就越容易发生）
13. 对于自己的文本向量，可以使用keras工具进行向量化
14. Embedding输入的是句子组成的向量，相当于一句一个一个向量，示例中，每个句子取20个单词，取所有词汇的 top 10000个单词来计算索引
15. 在我学习神经网络过程中，我有一种感觉，神经网络很像命令式编程，我定义好过程，不干涉具体执行
16. 神经网络的神奇之处在于，你只需要知道有什么最好，而不需要这个具体的东西，因为神经网络会帮你找到，你只需要知道你应该有这个东西，麻烦神经网络通过题海战术把他找到。
就RNN来说，我们认为应该需要有一个状态和输入结合起来，但是我们不确定，那我们就加入这个状态，然后让神经网络去训练就好了
17. 向量化：有些数据比如文本，不是数值型的，往往需要先把数据进行向量化
18. 画张量流向图的方式去思考
## 问题栏
1. 梯度消失问题以及数学原理
2. 神经网络每一层的参数是共享的么,还是说有的是有的不是
3. 我怎么知道可不可以预测呢？其实把时间作为一种属性，只要模型确定，那又有何不可呢？
## 错题本
1. pandas 基本以及弃用直接数组索引，必须使用iloc和loc，建议记住这种，不用数组索引
2. pandas 的轴跟numpy的轴的含义是相反的，pandas的0为列
3. extend 并不会返回一个值，只会在原有数据上进行修改
4. 弄清楚轴与行，axis0与列平行，垂直于行，axis1与行平行，垂直于列
5. std = np.std(train_data[:200000].astype(float), axis=0)  # 这里曾经以为类型不对应无法进行计算，保险起见最好转换一下或者检查一下
