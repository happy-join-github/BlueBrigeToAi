# 使用决策树进行调优

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
file = pd.read_csv('telecom_churn.csv')
# 处理这两列的数据变成整型。
file['International plan'] = pd.factorize(file['International plan'])[0]
file['Voice mail plan'] = pd.factorize(file['Voice mail plan'])[0]

file['Churn'] = file['Churn'].astype('int')

states = file['State']
y = file['Churn']

file.drop(['State','Churn'],axis=1,inplace=True)

x_train, x_holdout,y_train,y_holdout= train_test_split(file.values,y,test_size=0.3,random_state=17)



tree_params = {'max_depth':range(5,7),"max_features":range(16,18)}
tree = DecisionTreeClassifier(max_depth=5,random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)
tree_grid = GridSearchCV(tree,tree_params,cv=5,n_jobs=-1,verbose=True)

tree_grid.fit(x_train,y_train)

# {'max_depth': 6, 'max_features': 17}
# 0.94257014456259
# 0.946

print(tree_grid.best_params_)
print(tree_grid.best_score_)
print(accuracy_score(y_holdout,tree_grid.predict(x_holdout)))

