from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
train = open('news_train.txt','r',encoding='utf-8').read().splitlines()
label = open('label_newstrain.txt','r',encoding='utf-8').read().splitlines()
label = [int(i) for i in label]

test = open('news_test.txt','r',encoding='utf-8').read().splitlines()

model = CountVectorizer()
train_feature = model.fit_transform(train)
test_feature = model.transform(test)

clf = LogisticRegression()
score = cross_val_score(clf,train_feature,label,cv=5)
if score.mean()>=0.9:
    clf.fit(train_feature,label)
    pre = [str(i)+'\n' for i in clf.predict(test_feature)]
    with open('pred_test.txt','w',encoding='utf-8') as f:
        f.writelines(pre)
