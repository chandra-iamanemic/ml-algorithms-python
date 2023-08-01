#%% 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%%
X, y = datasets.make_regression(n_samples=200, n_features=1, noise=10, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)

#%%
fig = plt.figure()
plt.scatter(X[:,0], y, color="g", marker="x")


# %%
from LinearRegression import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)

def mean_squared_error(y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

mse = mean_squared_error(y_test, predictions)

# %%

lr_array = [0.0001, 0.001, 0.01, 0.1]

for lr in lr_array:
    
    model = LinearRegression(lr=lr)
    model.fit(X_train, y_train)
    fig = plt.figure()
    plt.scatter(X_train, y_train)
    plt.scatter(X_test, y_test)
    plt.plot(X, model.predict(X))
    plt.title(f"Learning Rate : {lr}")
    plt.show()

# %%
