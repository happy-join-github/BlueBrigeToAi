import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

file = pd.read_csv('telecom_churn.csv')
file['International plan'] = pd.factorize(file['International plan'])[0]
file['Voice mail plan'] = pd.factorize(file['Voice mail plan'])[0]

file['Churn'] = file['Churn'].astype('int')
states = file['State']
y = file['Churn']
file.drop(['State','Churn'],axis=1,inplace=True)

X_train, x_holdout,y_train,y_holdout= train_test_split(file.values,y,test_size=0.3,random_state=17)

# 这行代码创建了一个随机森林分类器实例。
# n_estimators=100 表示森林中包含100棵决策树，n_jobs=-1 表示使用所有可用的CPU核心进行并行计算，
# random_state=17 确保了随机森林初始化时的可重复性。
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1,random_state=17)

# 这行代码计算了随机森林分类器在训练集上的5折交叉验证的平均准确率。
# cross_val_score 函数对训练集进行交叉验证，cv=5 表示将数据集分成5份，轮流使用其中4份进行训练，1份进行验证。
np.mean(cross_val_score(forest, X_train, y_train, cv=5))

# 这行代码定义了随机森林分类器的参数搜索空间。
# max_depth 参数指定了树的最大深度，
# max_features 参数指定了在分割节点时考虑的最大特征数量，
forest_params = {'max_depth': range(8, 10),
                 'max_features': range(5, 7)}

# 这行代码创建了一个网格搜索实例，用于在定义的参数空间内搜索最佳参数组合。
# GridSearchCV 会评估参数组合的性能，并使用交叉验证（cv=5）来评估每个组合。
# n_jobs=-1 表示使用所有可用的CPU核心，verbose=True 表示在搜索过程中输出详细信息。
forest_grid = GridSearchCV(forest, forest_params,cv=5, n_jobs=-1, verbose=True)

forest_grid.fit(X_train, y_train)
# {'max_depth': 9, 'max_features': 6}
# 0.9511372931045574
# 0.953
print(forest_grid.best_params_)
print(forest_grid.best_score_)
print(accuracy_score(y_holdout,forest_grid.predict(x_holdout)))