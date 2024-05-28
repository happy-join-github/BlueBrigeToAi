from sklearn.feature_extraction.text import CountVectorizer

# 示例文本数据
texts = [
    '这 是 第一个 文档。',
    '这 是 第二个 文档。',
    '而 第三个 文档 与 前 两个 不同。'
]

# 初始化CountVectorizer
vectorizer = CountVectorizer()

# 拟合并转换文本数据
X = vectorizer.fit_transform(texts)

# 查看词汇表（即特征名称）
print("Vocabulary:", vectorizer.get_feature_names_out())

# 查看转换后的特征矩阵
print("Feature matrix:")
print(X.toarray())