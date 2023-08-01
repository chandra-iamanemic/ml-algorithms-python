#%% [markdown]
# In linear regression we modelled a continous output as a line y = wx + b
# In logistic regression we will want an output as probability of an input belonging to a class
# we model a probability by taking the y = wx + b equation and plugging it into a sigmoid function
# the y = wx + b line is now transformed into a range of 0 to 1 by our sigmoid function



#%%
import numpy as np

#%% 
class LogisticRegression:

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
            y_pred = self.sigmoid(np.dot(X, self.W) + self.b)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.W = self.W - self.lr * dw
            self.b = self.b - self.lr * db

    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.W) + self.b)
        return y_pred
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    

