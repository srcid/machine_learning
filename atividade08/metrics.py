import numpy as np

def mse(y_pred, y_true):
    return ((y_true - y_pred)**2).sum() / len(y_pred)

def rmse(y_pred, y_true):
  return np.sqrt(mse(y_pred, y_true))

def mae(y_pred, y_true):
  return np.abs(y_pred - y_true).sum() / len(y_pred)

def accuracy(y_pred, y_true):
  return np.sum(y_pred == y_true) / len(y_pred)