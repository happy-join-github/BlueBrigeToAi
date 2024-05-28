# tensorflow的优化，对初学者更加友好了

# 矩阵操作同numpy一样
import numpy as np
import tensorflow as tf
a = np.array([[1,2,3],[4,5,6]])

# nunpy转tensorflow
b = tf.convert_to_tensor(a)



c = tf.constant([[1, 2, 3],[4,5,6]])
d = tf.constant([[ 7, 8],[ 9 ,10],[11, 12]])

e = tf.matmul(c, d) # 矩阵相乘
print(e)
# 转置
e = tf.transpose(e)
print(e)