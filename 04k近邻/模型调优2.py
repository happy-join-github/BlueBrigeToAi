# 使用knn进行交叉验证

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

knn_pipe = Pipeline([('scaler',StandardScaler()),('knn',KNeighborsClassifier(n_jobs=-1))])

knn_params = {'knn__n_neighbors':range(6,8)}
knn_grid = GridSearchCV(knn_pipe,knn_params,cv=5,n_jobs=-1,verbose=True)

knn_grid.fit(x_train,y_train)

# {'knn__n_neighbors': 7}
# 0.8859867109023905
# 0.89

print(knn_grid.best_params_)
print(f"交叉验证的正确率{knn_grid.best_score_}")
print(f"模型的准确率{accuracy_score(y_holdout, knn_grid.predict(x_holdout))}")
