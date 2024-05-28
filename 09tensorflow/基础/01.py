import numpy as np
import tensorflow as tf
# 0阶张量(标量)
a = 1
# 1阶张量(一维数组)
a = np.array([1,2,3,4,5,6])

# 二阶张量(矩阵)
a = np.matrix(a).reshape((2,3))

# 三阶张量
a = np.array([[[1,2,3],[1,2,3]],[[4,5,6],[7,8,9]]])

b = tf.Variable([1,2,3,4,5,6])
c = tf.constant([4,5,6,7,8,9,])
# 查看数据
# print(b.numpy())
# print(c.numpy())

# 0 / 1 张量
b = tf.zeros((2,3))
c = tf.ones((3,2))

# 简单的0/1 张量创建
# 创建一个和d形状一样的张量
d = np.array([[1,2,3],[4,5,6]])
# 创建了一个shape = (2,3)的全零/全一矩阵
e = tf.zeros_like(d)
f = tf.ones_like(d)

# 标量矩阵创建
# 大小为(2,3)的6矩阵
g = tf.fill((2,3),6)

# 等差矩阵
h = tf.linspace(1,10,5)
# 数字序列
i = tf.range(1,10,2)