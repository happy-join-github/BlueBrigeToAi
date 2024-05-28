import pandas as pd

# 读取CSV文件
df = pd.read_csv('songs_origin.csv')
# 处理缺失值
df = df.fillna(df.mean())
# 删除异常值
df = df[(df['acousticness_yr'] >= 0) & (df['acousticness_yr'] <= 1)]
# 删除重复的行
df = df.drop_duplicates()
# 保存修改后的DataFrame到新的CSV文件
df.to_csv('songs_processed.csv',index=False)