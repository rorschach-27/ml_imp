#!/user/bin/env python
# coding=utf-8
"""
@project : ml_imp
@author  : zy
@file   : Perceptron.py
@ide    : PyCharm
@time   : 2022/7/4 16:26
"""
import numpy as np
import random


class Perceptron(object):
    def __init__(self,
                 max_iter=5000,
                 eta=1e-2, ):
        self.eta_ = eta
        self.max_iter_ = max_iter
        self._w = 0
        self._b = 0
        self._iter = 0

    def sign(self, x, w, b):
        y = np.dot(x, w) + b
        return y

    def duality_fit(self, X_train, y_train):
        Gram = np.zeros([len(X_train), len(y_train)])
        alphe = np.zeros(len(X_train))

        for i in range(0, len(X_train)):
            for j in range(0, len(y_train)):
                Gram[i][j] = X_train[i][0] * X_train[j][0] + X_train[i][1] * X_train[j][1]
        separated = False
        while not separated:
            self._iter+=1
            for i in range(0, len(X_train)):
                sum = 0
                for j in range(0, len(X_train)):
                    sum += alphe[j] * y_train[j] * Gram[j][i]
                if (sum + self._b) * y_train[i] <= 0:
                    alphe[i] += self.eta_
                    self._b += self.eta_ * y_train[i]
                    separated = False

        w = np.array([0., 0.])
        for i in range(0, len(alphe)):
            w[0] += alphe[i] * y_train[i] * X_train[i][0]
            w[1] += alphe[i] * y_train[i] * X_train[i][1]
        slope = -w[0] / w[1]
        intercept = -self._b / w[1]
        return slope, intercept

    def origin_fit(self, X_train, y_train):
        is_wrong = False
        self._w = np.zeros(len(X_train[0]), dtype=np.float)
        while not is_wrong:
            self._iter += 1
            wrong_count = 0
            for d in range(len(X_train)):
                X = X_train[d]
                y = y_train[d]
                if y * self.sign(X, self._w, self._b) <= 0:
                    self._w = self._w + self.eta_ * np.dot(y, X)
                    self._b = self._b + self.eta_ * y
                    wrong_count += 1
            if wrong_count == 0:
                is_wrong = True

    def predict(self, x, y):
        if self.sign(x, self._w, self._b) > 0:
            return 1
        else:
            return 0

    def get_iter(self):
        return self._iter


X_train = np.array([[3, 3], [4, 3], [1, 1]])
y_train = np.array([1, 1, -1])
model = Perceptron()
model.duality_fit(X_train, y_train)
x = np.array([5, 7])
kk = model.get_iter()
juyt = model.predict(x, np.array([1]))
print(0)
