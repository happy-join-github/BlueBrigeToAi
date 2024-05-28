# 导报
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

file=pd.read_csv('songs_train.csv')
x,y = file[file.columns.drop('popularity')].values,file['popularity'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
# model=DecisionTreeRegressor(max_leaf_nodes=5)
# 构建模型实例对象
model = DecisionTreeRegressor()
model.fit(x_train,y_train)
# 第一个参数真实值   第二个参数就是预测值
# score=r2_score(y_test,model.predict(x_test))

# 进行五折交叉验证
score=cross_val_score(model,x,y,cv=5)
# if score>=0.8:
#     f1 = pd.read_csv('songs_test.csv')
#     x = f1[f1.columns.drop('popularity')].values
#     f1['popularity'] = model.predict(x)
#     f1.to_csv('songs_testout.csv')
print(score)

# 决策树
# 随机森林



