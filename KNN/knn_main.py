#%%
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

#%%
iris_dataset = datasets.load_iris()
X, y = iris_dataset.data, iris_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)

#%%

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)

#%%

from KNN import KNN

model = KNN(k=5)

model.fit(X_train, y_train)
predictions = model.predict(X_test)


#%%

accurate_predictions = len(predictions == y_test)

accuracy = accurate_predictions/len(y_test)

# %%
