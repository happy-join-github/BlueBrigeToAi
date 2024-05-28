import numpy as np
from scipy.spatial.distance import euclidean

# 这一转换保留了时间点间的距离，对于需要估计距离的算法（kNN、SVM、k-均值等）使用这种方法来处理时间数据会很有帮助
def make_harmonic_features(value, period=24):
    value *= 2 * np.pi / period
    return np.cos(value), np.sin(value)


print(euclidean(make_harmonic_features(23), make_harmonic_features(1)))