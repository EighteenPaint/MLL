#!usr/bin/env python3
# -*- coding:utf-8 -*-
"""
#@author:Benny.Chen
#@file: 0503.py
#@time: 2020/6/22 20:50
#@email:chenbinkria@163.com
"""
from pprint import pprint

from keras.models import load_model
from keras.applications import VGG16

model = load_model('minst-model.h5')
# conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
# conv_base.summary()
pprint(model.layers)
