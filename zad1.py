import numpy as np
from NeuralNetwork import NeuralNetwork


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
