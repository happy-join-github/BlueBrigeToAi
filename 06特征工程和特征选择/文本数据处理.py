import numpy as np
def a():
    texts = ['i have a cat',
             'you have a dog',
             'you and i have a cat and a dog']
    # 分词
    vocabulary = list(enumerate(set([word for sentence in texts for word in sentence.split()])))
    print(vocabulary)
    
    # 统计词频
    def vectorize(text):
        vector = np.zeros(len(vocabulary))
        for i, word in vocabulary:
            num = 0
            for w in text:
                if w == word:
                    num += 1
            if num:
                vector[i] = num
        return vector
    
    for sentence in texts:
        print(vectorize(sentence))





def main():
    # 导入文本特征提取工具
    from sklearn.feature_extraction.text import CountVectorizer
    # 转换词矩阵 单个词到连续的两个词
    vect = CountVectorizer(ngram_range=(1, 2))
    vect.fit_transform(['no i have cows', 'i have no cows']).toarray()
    # 词汇表
    print(vect.vocabulary_)
# main()


def b():
    # 基于字符
    from scipy.spatial.distance import euclidean
    from sklearn.feature_extraction.text import CountVectorizer

    vect = CountVectorizer(ngram_range=(3, 3), analyzer='char_wb')

    n1, n2, n3, n4 = vect.fit_transform(
        ['andersen', 'petersen', 'petrov', 'smith']).toarray()
    
    # 查看欧式空间距离
    print(n1,n2,n3,n4)
    print(euclidean(n1, n2))
    print(euclidean(n2, n3))
    print(euclidean(n3, n4))
b()