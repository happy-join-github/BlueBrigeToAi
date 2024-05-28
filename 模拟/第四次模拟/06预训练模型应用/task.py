# 向量合成
# 顺序resnet、inception、xception
# MDAwMA

import json


all_train = []
all_train_label = []
with open('resnet_train.json', 'r', encoding='utf-8') as f1:
    with open('inception_train.json', 'r', encoding='utf-8') as f2:
        with open('xception_train.json', 'r', encoding='utf-8') as f3:
            f1dic = json.loads(f1.read())
            f2dic = json.loads(f2.read())
            f3dic = json.loads(f3.read())
            
            searchkey = [key for key in f1dic.keys()]
            for key in searchkey:
                tmp = []
                tmp += f1dic[key]['feature'] + f2dic[key]['feature'] + f3dic[key]['feature']
                # dic[key]={'feature':tmp,"label":f1dic[key]["label"]}
                all_train.append(tmp)
                all_train_label.append(f1dic[key]["label"])

# 构造分类模型进行分类
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(all_train, all_train_label, test_size=0.2)
rfc = RandomForestClassifier(n_estimators=500, verbose=1, n_jobs=-1,random_state=42)

rfc.fit(x_train, y_train)
score = accuracy_score(y_test, rfc.predict(x_test))
if score >= 0.95:
    print('score:', score)
    with open('result.csv', 'a', encoding='utf-8') as f:
        f.write('id,label\n')
        with open('test.json', 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            for key in json_data.keys():
                feature = json_data[key]['feature']
                label = rfc.predict([feature])
                f.write(f"{key},{label[0]}\n")
else:
    print('score:', score)
    print('正确率不足95%，请调整参数后在进行训练')