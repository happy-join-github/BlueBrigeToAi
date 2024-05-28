# 导库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# 准备数据集
file = pd.read_csv('songs_train.csv')

# 读取目标值和特征值
x,y = file[file.columns.drop('popularity')].values,file['popularity'].values

# 数据集划分
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

# 创建模型对象
model = DecisionTreeClassifier(max_depth=10)

# 开始训练
model.fit(x_train,y_train)

# 输出
feature_improtance = model.feature_importances_
feature_name = file.columns.drop('popularity')
# 创建dataframe
importance = pd.DataFrame({'feature':feature_name,'importance':feature_improtance})
# 降序排列
importance = importance.sort_values(by='importance',ascending=False)
# 提取特征和比重
dic = {key:val for key,val in zip(feature_name,feature_improtance) if key in [item[0] for item in importance.head(8).values]}
with open('decision.json','w',encoding='utf-8') as f:
    import json
    json.dump(dic,f,ensure_ascii=False)
    
# 全部特征的比重
# print(importance)

# 我们只取前八个特征
importance = importance.head(8)
# 随机森林分析出来的重要的八个特征
lst1 = ['popularity_ar', 'speechiness', 'loudness', 'acousticness', 'energy', 'popularity_yr', 'speechiness_yr', 'instrumentalness']
# 决策树重要特征列表
lst2 = [key[0] for key in importance.values]
print(lst2)


