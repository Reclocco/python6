import numpy as np


def relu(x):
    return x * (x > 0)


def relu_d(x):
    return 1.0 * (x > 0)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def sigm_d(x):
    return x * (1.0 - x)


class NeuralNetwork:
    def __init__(self, x, y, eta, seed):
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

    def feedforward(self, ftype_1, ftype_2):
        if ftype_1 == 0:
            self.layer1 = sigmoid(np.dot(self.input, self.weights1.T))
        else:
            self.layer1 = relu(np.dot(self.input, self.weights1.T))

        if ftype_2 == 0:
            self.output = sigmoid(np.dot(self.layer1, self.weights2.T))
        else:
            self.output = relu(np.dot(self.layer1, self.weights2.T))

    def backprop(self, ftype_1, ftype_2):
        if ftype_2 == 0:
            delta2 = sigm_d(self.output) * (self.y - self.output)
        else:
            delta2 = relu_d(self.output) * (self.y - self.output)

        d_weights2 = self.eta * np.dot(delta2.T, self.layer1)

        if ftype_1 == 0:
            delta1 = sigm_d(self.layer1) * np.dot(delta2, self.weights2)
        else:
            delta1 = relu_d(self.layer1) * np.dot(delta2, self.weights2)

        d_weights1 = self.eta * np.dot(delta1.T, self.input)

        self.weights1 += d_weights1
        self.weights2 += d_weights2


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

Y = np.array([[0], [1], [1], [0]])
np.set_printoptions(precision=3, suppress=True)

difference = []
print("testing for best xor, eta=", 0.09)
for i in range(2):
    for j in range(2):
        nn = NeuralNetwork(X, Y, 0.09, 1)

        for _ in range(5000):
            nn.feedforward(i, j)
            nn.backprop(i, j)

        if i == 0:
            ftype1 = "sigmoid"
        else:
            ftype1 = "relu"

        if j == 0:
            ftype2 = "sigmoid"
        else:
            ftype2 = "relu"

        print("1st layer:", ftype1, "2nd layer:", ftype2)
        print(nn.output)

print("\n\n")

############################################################################################################
#                                                    OR
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

Y = np.array([[0], [1], [1], [1]])

nn = NeuralNetwork(X, Y, 0.5, 0)
nn.feedforward(0, 0)
nn.backprop(0, 0)
print("OR:\n", nn.output, "\n")

############################################################################################################
#                                                    AND
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

Y = np.array([[0], [0], [0], [1]])

nn = NeuralNetwork(X, Y, 0.5, 0)
nn.feedforward(0, 0)
nn.backprop(0, 0)
print("AND:\n", nn.output)
