import numpy as np
from collections import Counter
from utils import *

class KNNClassifier:
  def __init__(self, k):
    self.k = k

  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

  def predict(self, X_test):
    predicted_labels = [self._predict(x) for x in X_test]

    return np.array(predicted_labels)

  def _predict(self, x):
    distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_labels = [self.y_train[i] for i in k_indices]
    most_commom = Counter(k_nearest_labels).most_common(1)[0][0]

    return most_commom

class KNNRegressor:
  def __init__(self, k):
    self.k = k

  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train

  def predict(self, X_test):
    predicted_labels = [self._predict(x) for x in X_test]

    return np.array(predicted_labels)

  def _predict(self, x):
    distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
    k_indices = np.argsort(distances)[:self.k]
    k_nearest_vals = [self.y_train[i] for i in k_indices]
    val = np.mean(k_nearest_vals)

    return val


class LinearRegression:
  def __init__(self, learning_rate, n_iters):
    self.learning_rate = learning_rate
    self.n_iters = n_iters
    self.weights = None
    self.bias = None

  def fit(self, X_train, y_train):
    n_samples, n_features = X_train.shape
    self.weights = np.zeros(n_features)
    self.bias = 0

    for _ in range(0,self.n_iters):
      y_pred = np.dot(X_train, self.weights) + self.bias
      dw = (1/n_samples) * np.dot(X_train.T, (y_pred - y_train))
      db = (1/n_samples) * np.sum(y_pred - y_train)

      self.weights = self.weights - self.learning_rate * dw
      self.bias = self.bias - self.learning_rate * db

  def predict(self, X_test):
    y_pred = np.dot(X_test, self.weights) + self.bias
    return y_pred