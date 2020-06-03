import numpy as np


class NeuralNetwork:
    def __init__(self, x, y, eta, seed=0):
        if seed == 1:
            np.random.seed(17)
        else:
            np.random.seed(None)

        self.input = x
        self.weights1 = np.random.rand(4, self.input.shape[1])
        self.weights2 = np.random.rand(1, 4)
        self.y = y
        self.layer1 = np.zeros((1, 4))
        self.output = np.zeros(self.y.shape)
        self.eta = eta

    def relu(self, x):
        return x * (x > 0)

    def relu_d(self, x):
        return 1.0 * (x > 0)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigm_d(self, x):
        return x * (1.0 - x)

    def feedforward(self, ftype_1, ftype_2):
        if ftype_1 == 0:
            self.layer1 = self.sigmoid(np.dot(self.input, self.weights1.T))
        else:
            self.layer1 = self.relu(np.dot(self.input, self.weights1.T))

        if ftype_2 == 0:
            self.output = self.sigmoid(np.dot(self.layer1, self.weights2.T))
        else:
            self.output = self.relu(np.dot(self.layer1, self.weights2.T))

    def backprop(self, ftype_1, ftype_2):
        if ftype_2 == 0:
            delta2 = self.sigm_d(self.output) * (self.y - self.output)
        else:
            delta2 = self.relu_d(self.output) * (self.y - self.output)

        d_weights2 = self.eta * np.dot(delta2.T, self.layer1)

        if ftype_1 == 0:
            delta1 = self.sigm_d(self.layer1) * np.dot(delta2, self.weights2)
        else:
            delta1 = self.relu_d(self.layer1) * np.dot(delta2, self.weights2)

        d_weights1 = self.eta * np.dot(delta1.T, self.input)

        self.weights1 += d_weights1
        self.weights2 += d_weights2
