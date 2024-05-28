import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

file = pd.read_csv('songs_train.csv')

x,y = file[file.columns.drop('popularity')].values,file['popularity'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = RandomForestRegressor(n_estimators=100)

model.fit(x_train,y_train)

feature_name = file.columns.drop('popularity')
feature_importance = model.feature_importances_

df = pd.DataFrame({'feature_name':feature_name,'importance':feature_importance})
df = df.sort_values(by='importance',ascending=False)
print(df)

# lst = [val[0] for val in df.head(8).values]
lst = ['popularity_ar', 'speechiness', 'loudness', 'acousticness', 'energy', 'popularity_yr', 'speechiness_yr', 'instrumentalness']

print(lst)




