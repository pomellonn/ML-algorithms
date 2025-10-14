import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

df = pd.read_csv('../dataframes/Housing.csv')
df.info()


def fit_my(X, y, lr=1e-2, max_iter=20000, tol=1e-8):
    n, features = X.shape
    weights = np.zeros(features, dtype=float)
    bias = 0.0
    prev_loss = np.inf

    for it in range(max_iter):
        y_pred = X @ weights + bias
        err = y_pred - y

        loss = (err @ err) / (2 * n)
        grad_b = err.mean()
        grad_w = (X.T @ err) / n

        weights -= lr * grad_w
        bias -= lr * grad_b

        if abs(prev_loss - loss) < tol or np.linalg.norm(np.concatenate((grad_w, [grad_b]))) < tol:
            break
        prev_loss = loss

    return weights, bias


def predict_my(X_test, w, b):
    return X_test @ w + b


def metrics(y, y_):
    N = len(y)
    mse = 0
    for i in range(N):
        mse += (y_[i] - y[i]) ** 2
    mse *= 1 / N

    mae = 0
    for i in range(N):
        mae += abs(y_[i] - y[i])
    mae *= 1 / N
    rmse = (mse) ** 0.5

    return mse, mae, rmse


features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
X = df[features].to_numpy(dtype=float)
y = df['price'].to_numpy(dtype=float)

X = scale(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

w, b = fit_my(X_train, y_train)
y_pred = predict_my(X_test, w, b)
mse, _, _ = metrics(y_test, y_pred)
print(f"My LinReg:\n MSE = {mse}")
print("b:", b)
print("w:", w)

sk_lin = LinearRegression().fit(X_train, y_train)
p = sk_lin.predict(X_test)
print(f"sklearn LinReg:\n MSE = {mean_squared_error(y_test, p)}")
