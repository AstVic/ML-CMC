import numpy as np

np.random.seed(42)

def gradient(y_true: int, y_pred: float, x: np.array) -> np.array:
    x_extended = np.append(x, 1)
    grad = (y_pred - y_true) * x_extended
    return grad

def update(alpha: np.array, gradient: np.array, lr: float):
    alpha_new = alpha - lr * gradient
    return alpha_new

def train(
    alpha0: np.array, x_train: np.array, y_train: np.array, lr: float, num_epoch: int
):
    alpha = alpha0.copy()
    for epo in range(num_epoch):
        for i, x in enumerate(x_train):
            x_extended = np.append(x, 1)
            z = np.dot(alpha, x_extended)
            y_pred = 1 / (1 + np.exp(-z))
            grad = gradient(y_train[i], y_pred, x)
            alpha = update(alpha, grad, lr)
    return alpha
