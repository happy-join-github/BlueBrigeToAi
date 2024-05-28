from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 加载数据集
with open('./news_train.txt', 'r', encoding='utf-8') as f:
    X_train = f.read().splitlines()

with open('./label_newstrain.txt', 'r', encoding='utf-8') as f:
    y_train = [int(label) for label in f.read().splitlines()]

with open('./news_test.txt', 'r', encoding='utf-8') as f:
    X_test = f.read().splitlines()

# 文本特征提取
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# 构建模型
clf = LogisticRegression()
clf.fit(X_train_counts, y_train)

# 预测测试集
y_pred = clf.predict(X_test_counts)

score = accuracy_score(y_train, y_pred)
if score >= 0.9:
    # 将预测结果保存到pred_test.txt
    with open('./pred_test.txt', 'w', encoding='utf-8') as f:
        for label in y_pred:
            f.write(str(label) + '\n')
else:
    print('no')

# 评估模型性能（仅在本地验证使用，实际提交时不计算准确率）
# 注意：这里只是示例，实际提交时不会提供真实标签
# 假设我们有一个真实的测试集标签 y_test 用于评估
# y_test = [...]  # 这里应该是真实的测试集标签
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model accuracy: {accuracy:.2f}")

# 请注意，由于题目要求中没有提供真实的测试集标签，所以上述评估代码块应被注释掉。
# 在实际提交时，只需要保证pred_test.txt文件的格式和内容正确即可。
