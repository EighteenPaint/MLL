## pandas基本数据结构 <br/>
1. DataFrame<br/>
2. Serices<br/>
## pandas索引<br/>
1. 默认索引是数字<br/>
2. 可以通过index参数和column参数指定行和列的参数<br/>
3. 可以通过数组和字典创建DataFrame对象，可以将其理解为数据库操作，而且跟spark数据处理接口有相似地方<br/>
## pandas常用操作<br/>
1. head或者使用数组切片方式都可以实现获取部分数据<br/>
2. loc():通过标签定位获取数据<br/>
3. iloc()：通过位置定位获取数据<br/>
4. 缺失数据的处理<br/>
    1. 删除dropna<br/>
    2. 填充fillna<br/>
5. 统计函数，均值，方差，中位数，众数....<br/>
6. join & merge<br/>
    1. join,简化了merge的行拼接操作，与数据库连表有些相似<br/>
    2. merge还可以实现列和行的合并，比join更加灵活
