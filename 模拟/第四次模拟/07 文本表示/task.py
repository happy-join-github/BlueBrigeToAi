import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

w2v_file_path = "word2vec_model.bin"
W2V_MODEL = Word2Vec.load(w2v_file_path)
W2V_SIZE = 100


def get_w2v(word):
    # TODO
    return W2V_MODEL.wv[word] if word in W2V_MODEL.wv else None


def get_sentence_vector(sentence):
    # TODO
    martix = [get_w2v(x) for x in sentence if x in W2V_MODEL.wv]
    return np.mean(martix, axis=0) if len(martix) else np.zeros(W2V_SIZE)


def get_similarity(array1, array2):
    array1_2d = np.reshape(array1, (1, -1))
    array2_2d = np.reshape(array2, (1, -1))
    similarity = cosine_similarity(array1_2d, array2_2d)[0][0]
    return similarity


def main():
    # 测试两个句子
    sentence1 = '我不喜欢看新闻。'
    sentence2 = '我觉得新闻不好看。'
    sentence_split1 = jieba.lcut(sentence1)
    sentence_split2 = jieba.lcut(sentence2)
    # 获取句子的句向量
    sentence1_vector = get_sentence_vector(sentence_split1)
    sentence2_vector = get_sentence_vector(sentence_split2)
    # 计算句子的相似度
    similarity = get_similarity(sentence1_vector, sentence2_vector)
    print(similarity)


if __name__ == '__main__':
    main()
