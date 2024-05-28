import os

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
import keras.models


digits = load_digits()
x = digits.data
y = digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, random_state=1)

# 每一个epoch之后会输出在训练集和测试集上的分类评估结果
model = keras.models.Sequential()
# adam 优化器 + 交叉熵损失 + 准确度评估
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=50,validation_data=(x_test,y_test))
if not os.path.exists('./weights'):
    os.mkdir('./weights')
model.save('./model.h5')


