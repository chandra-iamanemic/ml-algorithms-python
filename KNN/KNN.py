#%%
import numpy as np
from collections import Counter


#%%

def calculate_euclidean_distance(a,b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a-b)**2))

#%%
class KNN():

    def __init__(self, k=3):
        self.k = k 
    

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        
        predictions = []

        for x in X:

            distances = [calculate_euclidean_distance(x, x_train) for x_train in self.X_train ]

            #Sort the distances and get their indices 

            distances_sorted = np.argsort(distances)

            # take the nearest k distances
            k_nearest_distances = distances_sorted[:self.k]

            k_nearest_labels = [self.y_train[i] for i in k_nearest_distances]


            # find the most number of labels to find the class of new input X

            highest_class_count = Counter(k_nearest_labels).most_common(1)[0][0]


            predictions.append(highest_class_count)

        return np.array(predictions)




