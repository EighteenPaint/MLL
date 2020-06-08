#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: TreeNode.py
#@time: 2020/6/8 9:39
#@email:chenbinkria@163.com
"""


class TreeNode:
    def __init__(self, feat, val, right, left):
        featureToSplitOn = feat
        valueOfSplit = val
        rightBranch = right
        leftBranch = left
