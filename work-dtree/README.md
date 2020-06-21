# 关于交叉验证<br/>
[Sklearn 交叉验证](https://blog.csdn.net/rocling/article/details/93717335?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-1)
# 作业记录
* 2020年6月20日11:06:49<br/>
选用简单决策树进行决策树构造，忽略了属性为连续值得情况，ID3算法只适合离散数据进行<br/>
处理方法思考：
  1. 将数值标准化，离散化，但是离散化后，如果是01化，显然会让很多重要的特征失去较多信息，如果分组，特征较多，处理起来也不方便<br/>
  2. 使用连续回归树或者模型树(可以不考虑)
## 思路<br/>
选用回归树的方式，采用CART算法，不过在叶子节点处需要进行处理，采用投票极值或者01函数进行处理
# 错题本<br/>
1. [0：-1]和[:,-1]是不一样的
2. SK的运算速度很快，很神奇,也许是底层做了优化
  