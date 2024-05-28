# 自动微分
import tensorflow as tf
w = tf.Variable([2.0])

with tf.GradientTape() as tape:
    loss = w * w

grad = tape.gradient(loss, w)
print(grad)
