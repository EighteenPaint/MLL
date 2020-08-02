#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: sample.py
#@time: 2020/7/27 10:17
#@email:chenbinkria@163.com
"""
from scipy.stats import uniform
# print(uniform.rvs(size=1000))
from scipy.stats import norm

x_star = norm.rvs(loc=1, scale=1, size=3)
print("star:", x_star)
amount = norm.pdf(x_star, loc=1, scale=1)
print("probability:", amount)
print("amount:",sum(amount))
