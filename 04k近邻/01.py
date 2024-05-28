# 使用决策树和k紧邻预测 客户离网率
import pandas as pd
file = pd.read_csv('telecom_churn.csv')
# 处理这两列的数据编程整型。
file['International plan'] = pd.factorize(file['International plan'])[0]
file['Voice mail plan'] = pd.factorize(file['Voice mail plan'])[0]

file['Churn'] = file['Churn'].astype('int')

states = file['State']
y = file['Churn']

file.drop(['State','Churn'],axis=1,inplace=True)

# 将数据集的 70% 划分为训练集，30% 划分为留置集。
# 留置集的数据在调优模型参数时不会被用到，在调优之后，用它评定所得模型的质量。

# 训练两个模型
# 一开始，我们并不知道如何设置模型参数能使模型表现好，所以可以使用随机参数方法，
# 假定树深（max_dept）为 5，近邻数量（n_neighbors）为 10。
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,StratifiedKFold

#划分数据集

x_train, x_holdout,y_train,y_holdout= train_test_split(file.values,y,test_size=0.3,random_state=17)

tree = DecisionTreeClassifier(max_depth=5,random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)
tree.fit(x_train,y_train)
knn.fit(x_train,y_train)

from sklearn.metrics import accuracy_score
# 决策树的正确率:0.94
# k近邻的正确率:0.881

print(f'决策树的正确率:{accuracy_score(y_holdout, tree.predict(x_holdout))}')
print(f'k近邻的正确率:{accuracy_score(y_holdout, knn.predict(x_holdout))}')