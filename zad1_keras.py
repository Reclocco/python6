from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy as np


model = Sequential()
model.add(Dense(units=4, activation='sigmoid', input_dim=3))
model.add(Dense(units=1, activation='sigmoid'))
print(model.summary())
print('\n')

sgd = optimizers.SGD(lr=1)
model.compile(loss='mean_squared_error', optimizer=sgd)

X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

Z = np.array([[0, 0, 0],
              [1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

Y = np.array([[0], [0], [0], [1]])
model.fit(X, Y, epochs=5000, verbose=True)
print(model.predict(Z))
