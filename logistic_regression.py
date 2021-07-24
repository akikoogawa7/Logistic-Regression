#%%
from numpy import load
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
            loss = self._compute_bce_loss(y_hat, y)
            grad_w, grad_b = self._compute_gradient(X, y)
            self.w -= lr * grad_w
            self.b -= lr * grad_b
            all_losses.append(loss)
        print('loss:', all_losses)
        plot_loss(all_losses)
    
    def _negative_sigmoid(self, z):
        self.exp = np.exp(z)
        return self.exp / (self.exp + 1)

    def _positive_sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid(self, z):
        positive = z >= 0
        # Boolean array inversion is faster than another comparison
        negative = ~positive

        # empty contains junk hence will be faster to allocate than zeros
        result = np.empty_like(z)
        result[positive] = self._positive_sigmoid(z[positive])
        result[negative] = self._negative_sigmoid(z[negative])
        return result

    def predict(self, X):
        return self._sigmoid(self.predict(X))

    def _bce(self, y_hat, y):
        y_hat = self.predict(X)
        loss = -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return np.mean(loss)

    def _compute_gradient(self):
        pass

def plot_loss(losses):
    plt.figure()
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.plot(losses)
    plt.show()

