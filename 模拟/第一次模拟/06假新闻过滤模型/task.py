import numpy as np
# 加载停用词
stop_words =[line.strip() for line in  open('study/stopwords.txt','r',encoding='utf-8').readlines()]

# 加载训练文件，对训练文件进行预处理
crops = []
label = [line.strip() for line in open('label_newstrain.txt','r',encoding='utf-8').readlines()]
label = np.array(label)

trainFile =[line.strip() for line in open('news_train.txt','r',encoding='utf-8').readlines()]

# 去掉停用词，不用分词因为源文件已经做好了分词
for line in trainFile:
    line = line.split()
    tmp = []
    for word in line:
        if word not in stop_words:
            tmp.append(word)
    crops.append(tmp)


# 特征提取
from gensim.models import Word2Vec
# 加载嵌入词
model = Word2Vec(crops)
# 计算特征向量
# 将每条数据各单词的向量相加，作为该句子最终的向量表示。
def sum_vec(text):
    vec = np.zeros(100).reshape((1,100))
    for word in text:
        if word in list(model.wv.key_to_index):
            vec +=model.wv[word].reshape((1,100))
    return vec

# 计算每个句子的向量
x = np.concatenate([sum_vec(z) for z in crops])
# print(x)
def run():
    # 开始训练
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    x_train,x_test,y_train, y_test = train_test_split(x,label,test_size=0.2,random_state=20)
    
    rfc =RandomForestClassifier()
    rfc.fit(x_train,y_train)
    print(rfc.score(x_test, y_test))
    
run()