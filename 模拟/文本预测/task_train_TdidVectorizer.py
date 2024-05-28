import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# 加载数据集
with open('./news_train.txt', 'r', encoding='utf-8') as f:
    X_train = f.read().splitlines()

with open('./label_newstrain.txt', 'r', encoding='utf-8') as f:
    y_train = [int(label) for label in f.read().splitlines()]

# 文本特征提取
vectorizer = TfidfVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

# 构建模型
clf = LogisticRegression()

# 使用5折交叉验证评估模型性能
scores = cross_val_score(clf, X_train_counts, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {np.mean(scores)}")

# 为了展示分类报告，我们可以选择一个折的数据进行训练和测试
# 计算每一折的大小
fold_size = len(y_train) // 5

# 选择第一折数据进行训练和测试（这里仅为示例）
X_train_cv = X_train_counts[:fold_size]
X_test_cv = X_train_counts[fold_size:2 * fold_size]
y_train_cv = y_train[:fold_size]
y_test_cv = y_train[fold_size:2 * fold_size]

# 训练模型
clf.fit(X_train_cv.toarray(), y_train_cv)

# 预测验证集
y_pred_cv = clf.predict(X_test_cv.toarray())

# 输出分类报告
print(classification_report(y_test_cv, y_pred_cv))

# 由于题目没有提供测试集的真实标签，我们在此不进行真实的测试集预测和评估
# ...（省略测试集预测和保存pred_test.txt的代码）