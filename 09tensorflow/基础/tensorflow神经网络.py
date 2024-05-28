import tensorflow as tf
import pandas as pd


class Model(object):
    def __init__(self):
        self.W1 = tf.Variable(tf.ones([2, 3]))
        self.W2 = tf.Variable(tf.ones([3, 1]))

    def __call__(self, x):
        hidden_layer = tf.nn.sigmoid(tf.linalg.matmul(X, self.W1))
        y_ = tf.nn.sigmoid(tf.linalg.matmul(hidden_layer, self.W2))
        return y_

def loss_fn(model, X, y):
    y_ = model(X)
    loss = tf.reduce_mean(tf.losses.mean_squared_error(y_true=y, y_pred=y_))
    return loss

df = pd.read_csv('course-12-data.csv')
X = tf.cast(tf.constant(df[['X0', 'X1']].values), tf.float32)
y = tf.constant(df[['Y']].values)

model = Model()
EPOCHS = 200

for epoch in range(EPOCHS):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, X, y)
    grads = tape.gradient(loss, [model.W1, model.W2])
    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    optimizer.apply_gradients(zip(grads, [model.W1, model.W2]))
    
file = pd.read_csv('course-12-data.csv')
X = tf.cast(tf.constant(file[['X0', 'X1']].values), tf.float32)
y = tf.constant(file[['Y']].values)

# 我们仅需要构建一个神经网络前向传播图，训练时就能够实现自动求导并更新参数。
model =Model()
y_ = model(X) #测试输入
print(y_.shape)


