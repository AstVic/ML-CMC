import numpy as np
np.random.seed(42)

class LinearRegression:
    def __init__(self, **kwargs):
        self.coef_ = None
        pass

    def fit(self, x: np.array, y: np.array):
        x_with_intercept = np.hstack([np.ones((x.shape[0], 1)), x])
        self.coef_ = np.linalg.pinv(x_with_intercept.T @ x_with_intercept) @ x_with_intercept.T @ y
        pass

    def predict(self, x: np.array):
        x_with_intercept = np.hstack([np.ones((x.shape[0], 1)), x])
        return x_with_intercept @ self.coef_
        pass
