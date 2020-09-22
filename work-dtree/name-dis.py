import datetime
from concurrent.futures import thread
from threading import Thread

import pandas as pd


def loadData(path):
    df = pd.read_json(path)
    return df


#
def Keywords(data):
    keywords = data.iloc[3, :]
    # corpus = []
    resultKeyWords = []
    for keyword in keywords.values:
        if isinstance(keyword, list):
            # corpus.extend(keyword)
            resultKeyWords.append(keyword)
        else:  # necessary and or the position/label will error
            keyword = ['']
            resultKeyWords.append(keyword)
            # corpus.extend(keyword)
    # with open("corpus", "w") as file:
    #     file.write(" ".join(corpus))
    # with open("keywords", "w") as file:
    #     for keyword in resultKeyWords:
    #         file.writelines(keyword)
    return resultKeyWords


data = loadData("thin_datas")
keywords = Keywords(data)
with open("corpus") as file:  # 使用保存的语料库，不用反复读取文件操作，提高效率
    corpus = file.read()
# fit on corpus
from sklearn.feature_extraction.text import CountVectorizer

vecizer = CountVectorizer(stop_words='english', token_pattern=r"(?u)\b\S+\b")  # 由于专业名词的原因，最好不要使用默认的token pattern
vecizer.fit([corpus])  # 进行训练

key_word_vector = []
import numpy as np

for keyword in keywords[0:100]:
    wordFlatten = [" ".join(keyword)]
    vetctor = vecizer.transform(wordFlatten).toarray()[0]
    temp = np.sum(vetctor)
    key_word_vector.append(vetctor)
from sklearn.cluster import DBSCAN

# 使用dbscan进行聚类
start = datetime.datetime.now()
# 观察半径
from sklearn.metrics.pairwise import euclidean_distances

# distance = euclidean_distances([vet for vet in key_word_vector])

# dis_df = pd.DataFrame(distance)

# print(dis_df)
stat = dict()


def dbscan(eps, min_sample, data, stat):
    print("new thread starting")
    y_pred = DBSCAN(eps=eps, min_samples=min_sample).fit_predict(data)
    print("result:", np.shape(y_pred), y_pred)
    stat[eps] = np.max(y_pred)
    stat_df = pd.DataFrame(stat)
    print(stat_df)


for eps in np.arange(start=1, step=0.2, stop=5):
    print("eps:", eps)
    # dbscan(eps, 5, key_word_vector, stat)
    y_pred = DBSCAN(eps=eps, min_samples=5).fit_predict(key_word_vector)
    end = datetime.datetime.now()
    # print("finished:cost time:", (end - start).seconds)
    print("result:", np.shape(y_pred), y_pred)
    stat[eps] = np.max(y_pred)
stat_df = pd.DataFrame(stat)
print(stat_df)
end = datetime.datetime.now()
# print("finished:cost time:", (end - start).seconds)
# while True:
#     pass
    # stat_df = pd.DataFrame(stat)
    # print(stat_df)
# get the cluster index
# 生成更加全面的语料库
