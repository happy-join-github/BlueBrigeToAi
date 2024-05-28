import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf

digits = load_digits()

x = digits.data
y = digits.target

# 数据预处理
y = np.eye(10)[y.reshape(-1)]

# 数据切分
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=.8, random_state=1)


class Model:
    def __init__(self):
        # 定义前向传播过程
        # 全连接1
        self.w1 = tf.Variable(tf.random.normal([64, 30]))
        self.b1 = tf.Variable(tf.random.normal([30]))
        # 全连接2
        self.w2 = tf.Variable(tf.random.normal([30, 10]))
        self.b2 = tf.Variable(tf.random.normal([10]))
    
    def __call__(self, x):
        x = tf.cast(x, tf.float32)
        
        fc1 = tf.nn.relu(tf.add(tf.matmul(x,self.w1),self.b1))
        
        fc2 = tf.nn.relu(tf.matmul(fc1,self.w2),self.b2)
        return fc2
def loss_fn(model,x,y):
    preds = model(x)
    # 使用热编码
    # 交叉熵损失函数+softmax的结合版本
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=preds,labels=y))
    return loss


def accuracy_fn(logits, labels):
    preds = tf.argmax(logits, axis=1)  # 取值最大的索引，正好对应字符标签
    labels = tf.argmax(labels, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


EPOCHS = 500
learn_rate = 0.02
model = Model()
for epoch in range(EPOCHS):
    with tf.GradientTape() as tape:
        loss = loss_fn(model,x_train,y_train)
    #需要优化参数列表
    train_variables = [model.w1,model.w2,model.b1,model.b2]
    # 求微分
    grads = tape.gradient(loss,train_variables)
    
    # 使用adam优化器
    optimizer = tf.optimizers.Adam(learning_rate=learn_rate)
    optimizer.apply_gradients(zip(grads,train_variables)) # 更新梯度

    accuracy = accuracy_fn(model(x_test), y_test)

    # 每 100 个 Epoch 输出各项指标
    if epoch == 0:
        print(f'Epoch [000/{EPOCHS}], Accuracy: [{accuracy:.2f}], Loss: [{loss:.4f}]')
    elif (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Accuracy: [{accuracy:.2f}], Loss: [{loss:.4f}]')