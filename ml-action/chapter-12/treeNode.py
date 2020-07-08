#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: treeNode.py
#@time: 2020/6/11 9:38
#@email:chenbinkria@163.com
"""


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        """
        输出树结构
        :param ind:
        :return:
        """
        print("     " * ind, self.name, "  ", self.count)
        for child in self.children.values():
            child.disp(ind + 1)
