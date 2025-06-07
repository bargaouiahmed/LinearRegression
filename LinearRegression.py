import numpy as np

from sklearn.preprocessing import StandardScaler


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.loss_hist = []
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X, y):
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y).ravel()

        n_samples, n_features = X_scaled.shape

        np.random.seed(42)
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0

        for i in range(self.n_iter):
            y_pred = np.dot(X_scaled, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X_scaled.T, (y_pred - y_scaled))
            db = (1 / n_samples) * np.sum(y_pred - y_scaled)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            loss = np.mean((y_pred - y_scaled) ** 2)
            self.loss_hist.append(loss)

            # Check for numerical issues
            if np.isnan(loss) or np.isinf(loss):
                print(f"Numerical overflow at iteration {i}. Try reducing learning rate.")
                break

    def predict(self, X):
        if hasattr(X, 'values'):
            X = X.values

        X_scaled = self.scaler_X.transform(X)

        y_pred_scaled = np.dot(X_scaled, self.weights) + self.bias

        y_pred_scaled = y_pred_scaled.reshape(-1, 1)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).ravel()

        return y_pred