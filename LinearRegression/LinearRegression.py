#%% [markdown]
# Fitting a curve to a continuous function 
# we use a linear function to fit given set of data
# we use y = wx + b
# we find the best possible w and b which give us the minimum value for our cost function
# Cost Function --> MSE --> Mean Squared Error (sum of the squared differences of predicted and actual) averaged over the number of samples
# How do we find the minimum of the cost function? We take its gradient and find the values of w and b that make it zero
# We do a gradient descent to arrive at the optimum values for weights and bias

#%%
import numpy as np

#%% 
class LinearRegression:

    def __init__(self, lr=0.001, n_iterations=1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.W = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.W = np.zeros(n_features)
        self.b = 0


        for i in range(self.n_iterations):
            # y_pred = wx + b
            y_pred = np.dot(X, self.W) + self.b
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.W = self.W - self.lr * dw
            self.b = self.b - self.lr * db

    def predict(self, X):
        y_pred = np.dot(X, self.W) + self.b
        return y_pred