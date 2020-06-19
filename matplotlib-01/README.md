# Matplotlib<br/>
1. 在代码模式下需要使用show方法，但是在交互式环境下，用plot就会画出<br/> 
## 折线图篇
0. xticks,yticks
1. rc进行更多matplot更多配置项
2. figure全局配置
3. font_manager
4. plt.xlabel，plt.ylabel，plt.title
5. fontproperity=自己设置的字体
6. plt.grid:设置网格<br/>
   alpha：0.4 透明度<br/>
   linestyle：网格样式<br/>
7. plt.legend,有多个数据在同一图表里显示时用，用于区分
8. plt.plot：<br/>
         x：x轴数据<br/>
         y：y轴数据<br/>
         label：标签<br/>
         color：图表颜色<br/>
         linestyle：线的style，实线或者虚线<br/>
         alpha<br/>
         linewidth：线宽<br/>
10. plt.savefig():保存为图片

## 散点图篇
plt.scatter:
    x
    y
## 条形图篇
plt.bar/barh

## 直方图篇
plt.hist:直方图是一维数据频率展示的图表

## 多图绘画
```python
fig = plt.figure()  
  
ax1 = fig.add_subplot(221)  #分成2行2列，选择第一个
ax1.plot(x, x)  #画第1个图
  
ax2 = fig.add_subplot(222)  #分成2行2列，选择第二个
ax2.plot(x, -x)  #画第2个图
  
ax3 = fig.add_subplot(223)  #分成2行2列，选择第三个
ax3.plot(x, x ** 2)  #画第3个图
  
ax4 = fig.add_subplot(224)  #分成2行2列，选择第四个
ax4.plot(x, np.log(x))  #画第4个图
  
plt.show()  
```
        
   