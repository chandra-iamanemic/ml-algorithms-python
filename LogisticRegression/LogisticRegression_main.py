#%% [markdown]
# In linear regression we modelled a continous output as a line y = wx + b
# In logistic regression we will want an output as probability of an input belonging to a class
# we model a probability by taking the y = wx + b equation and plugging it into a sigmoid function
# the y = wx + b line is now transformed into a range of 0 to 1 by our sigmoid function


#%% 
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#%%
dataset = datasets.load_breast_cancer()
X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 1234)

#%%
fig = plt.figure()
plt.scatter(X[:,0], y, color="g", marker="x")


# %%
from LogisticRegression import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)
predictions_probabilities = model.predict(X_test)

predictions = [1 if i>0.5 else 0 for i in predictions_probabilities]


def accuracy(y_true, y_predicted):
    return np.sum((y_true == y_predicted)/len(y_true))


acc = accuracy(y_test, predictions)
print(acc)



# %%
