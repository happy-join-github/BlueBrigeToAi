import numpy as np

from sklearn.datasets import load_digits
# 导入随机森林
from sklearn.ensemble import RandomForestClassifier
# 标准化特征的一个转换器，对特征减去其均值并除以其标准差，使得每个特征都有均值为0，标准差为1
from sklearn.preprocessing import StandardScaler
# 一个将多个步骤封装为一个单一对象的工具，可以用来序列化多个数据转换步骤与一个估算器。
from sklearn.pipeline import Pipeline
# 分割数据集
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
# 决策树和k近邻
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



data = load_digits()
X, y = data.data, data.target

X[0, :].reshape([8, 8])

    
x_train, x_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3, random_state=17)

tree = DecisionTreeClassifier(max_depth=5,random_state=17)

knn_pipe = Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_neighbors=10))])

tree.fit(x_train,y_train)
knn_pipe.fit(x_train,y_train)

tree_pred = tree.predict(x_holdout)

knn_pred = knn_pipe.predict(x_holdout)
# (0.666,0.976)
print(accuracy_score(y_holdout, tree_pred))
print(accuracy_score(y_holdout, knn_pred))


# 使用交叉验证检验决策树的的准确率
# tree_params = {'max_depth':[10,20,30],"max_features":[30,50,64]}
# tree_grid = GridSearchCV(tree,tree_params,cv=5,n_jobs=-1,verbose=True)
# tree_grid.fit(x_train,y_train)
# 0.8568203376968316
# {'max_depth': 10, 'max_features': 50}
# print(tree_grid.best_score_)
# print(tree_grid.best_params_)

# 使用交叉验证调优knn模型
# 0.9864858028204642
# print(np.mean(cross_val_score(KNeighborsClassifier(n_neighbors=1), x_train, y_train, cv=5)))

# 使用随机森林优化模型
# 0.9753462341111744
print(np.mean(cross_val_score(RandomForestClassifier(random_state=17), x_train, y_train, cv=5)))

# 从这个任务中得到的结论（同时也是一个通用的建议）：首先查看简单模型（决策树、最近邻）在你的数据上的表现，因为可能仅使用简单模型就已经表现得足够好了。