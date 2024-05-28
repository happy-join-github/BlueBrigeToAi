import numpy as np

def sigmoid(x):
    """激活函数"""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """sigmoid 函数求导"""
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, X, y, lr):
        """初始化参数"""
        self.input_layer = X
        self.W1 = np.ones((self.input_layer.shape[1], 3))  # 初始化权重全为 1
        self.W2 = np.ones((3, 1))
        self.y = y
        self.lr = lr

    def forward(self):
        """前向传播"""
        self.hidden_layer = sigmoid(np.dot(self.input_layer, self.W1))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.W2))
        return self.output_layer

    def backward(self):
        """反向传播"""
        d_W2 = np.dot(self.hidden_layer.T, (2 * (self.output_layer - self.y) *
                                            sigmoid_derivative(np.dot(self.hidden_layer, self.W2))))

        d_W1 = np.dot(self.input_layer.T, (
            np.dot(2 * (self.output_layer - self.y) * sigmoid_derivative(
                   np.dot(self.hidden_layer, self.W2)), self.W2.T) * sigmoid_derivative(
                np.dot(self.input_layer, self.W1))))

        # 参数更新
        self.W1 -= self.lr * d_W1
        self.W2 -= self.lr * d_W2