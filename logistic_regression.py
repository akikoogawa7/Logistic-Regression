#%%
from numpy import load
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.8)
X_train, y_train, X_val, y_val = train_test_split(X, y, test_size=0.8)

X.shape

df = pd.DataFrame(X)
data = load_breast_cancer()
data.keys()
column_names = load_breast_cancer()['feature_names']
df.columns = column_names

print(df)

class LogisticRegression:
    def __init__(self, n_features):
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

    def fit(self, X, y, epochs=10):
        lr = 0.001
        all_losses = []
        for epoch in range(epochs):
            y_hat = self.predict(X)
            loss = self.__compute_bce_loss__(y_hat, y)
            grad_w, grad_b = self.__compute_gradient__(X, y)
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            all_losses.append(loss)

    def predict(self, X):
        return np.matmul(X, self.w) + self.b

    def __compute_bce_loss__(self, y_hat, y):
        pass

    def __compute_gradient__(self):
        pass
