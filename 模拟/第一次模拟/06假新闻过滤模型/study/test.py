# 加载数据集
import pandas as pd

file = pd.read_csv('wsdm_mini.csv')
file['title_zh'] = file[['title1_zh','title2_zh']].apply(lambda x:''.join(x),axis=1)
# print(file['title1_zh'].head())
file_merge = file.drop(file.columns[[0,1]],axis=1)
# print(file_merge.head())

def load_stopword(filepath):
    with open(filepath,'r',encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    return stopwords

stop_words = load_stopword('stopwords.txt')

import jieba
from tqdm import tqdm
corpus = []
for line in tqdm(file['title_zh']):
    words = []
    seglst = list(jieba.cut(line))
    
    for word in seglst:
        if word not in stop_words:
            words.append(word)
    
    corpus.append(words)
    
# print(corpus)
# corpus 二维数组里面存放了处理后的数据，去掉停用词和标点符号的分词
# 特征提取
import numpy as np
from gensim.models import Word2Vec

model = Word2Vec(corpus)
def sum_vec(text):
    vec = np.zeros(100).reshape((1,100))
    for word in text:
        if word in list(model.wv.key_to_index):
            vec +=model.wv[word].reshape((1,100))
    return vec

X = np.concatenate([sum_vec(z) for z in tqdm(corpus)])

# 训练文本
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test = train_test_split(X,file['label'],test_size=0.2)

clf = RandomForestClassifier()
clf.fit(x_train,y_train)
print(clf.score(x_test, y_test))

