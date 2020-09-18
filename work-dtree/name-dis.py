import datetime

import pandas as pd


def loadData(path):
    df = pd.read_json(path)
    return df


def genCorpusAndKeywords(data):
    keywords = data.iloc[3, :]
    corpus = []
    resultKeyWords = []
    for keyword in keywords.values:
        if isinstance(keyword, list):
            corpus.extend(keyword)
            resultKeyWords.append(keyword)
        else:  # necessary and or the position/label will error
            keyword = ['']
            resultKeyWords.append(keyword)
            corpus.extend(keyword)
    return corpus, resultKeyWords


data = loadData("thin_datas")
corpus, keywords = genCorpusAndKeywords(data)
# fit on corpus
from sklearn.feature_extraction.text import CountVectorizer

vecizer = CountVectorizer(stop_words='english', token_pattern=r"(?u)\b\S+\b")  # 由于专业名词的原因，最好不要使用默认的token pattern
vecizer.fit(corpus)  # 进行训练

key_word_vector = []
import numpy as np

for keyword in keywords:
    wordFlatten = [" ".join(keyword)]
    vetctor = vecizer.transform(wordFlatten).toarray()[0]
    temp = np.sum(vetctor)
    key_word_vector.append(vetctor)
from sklearn.cluster import DBSCAN

start = datetime.datetime.now()
y_pred = DBSCAN(eps=3.8, min_samples=5).fit_predict(key_word_vector)
end = datetime.datetime.now()
print("finished:cost time:", (end - start).seconds)
print("result:", np.shape(y_pred), y_pred)

# get the cluster index
