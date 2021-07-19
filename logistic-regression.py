#%%
from numpy import load
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd

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
