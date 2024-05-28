from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import decomposition
import numpy as np


from sklearn.tree import DecisionTreeClassifier  # 导入决策树模型、
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score  # 识别准确率计算函数
iris = datasets.load_iris()
X = iris.data
y = iris.target
def yuanshi():
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        stratify=y,
                                                        random_state=42)
    
    # 决策树的深度设置为 2
    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)
    print('Accuracy: {:.5f}'.format(accuracy_score(y_test,
                                                   preds.argmax(axis=1))))

def solution():
    pca = decomposition.PCA(n_components=2)
    X_centered = X - X.mean(axis=0)
    pca.fit(X_centered)
    X_pca = pca.transform(X_centered)

    # 可视化 PCA 降维后的结果
    def show():
        plt.plot(X_pca[y == 0, 0], X_pca[y == 0, 1], 'bo', label='Setosa')
        plt.plot(X_pca[y == 1, 0], X_pca[y == 1, 1], 'go', label='Versicolour')
        plt.plot(X_pca[y == 2, 0], X_pca[y == 2, 1], 'ro', label='Virginica')
        plt.legend(loc=0)
        plt.show()

    # 训练集合测试集同时使用 PCA 进行降维
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=.3,
                                                        stratify=y,
                                                        random_state=42)

    clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict_proba(X_test)
    print('Accuracy: {:.5f}'.format(accuracy_score(y_test,
                                                   preds.argmax(axis=1))))
solution()