import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

file = pd.read_csv('songs_train.csv')
x,y = file[file.columns.drop(['popularity'])].values,file['popularity'].values
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = RandomForestRegressor()
model.fit(x_train,y_train)
score = r2_score(y_test,model.predict(x_test))
if score>=0.8:
    test_file = pd.read_csv('songs_test.csv')
    test_x= test_file[test_file.columns.drop('popolarity')].values
    test_file['popularity'] = model.predict(test_x)
    test_file.to_csv('songs_testout.csv',index=False)
print(score)