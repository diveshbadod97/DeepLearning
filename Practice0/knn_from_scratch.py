# -*- coding: utf-8 -*-

import numpy as np
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn.datasets import load_iris


def euc_dist(x1, x2):
    '''
    Parameters
    ----------
    x1 : Vector
        DESCRIPTION: A data point from the testing set
    x2 : Vector
        DESCRIPTION: A data point from the training set.

    Returns
    -------
    The Euclidean distance between the 2 vectors passed to the function.
    
    Task: Given a datapoint from the test set, this function caluclates the euclidean distance between this datapoint and each of
    datapoints in the training set one at a time (refer to the call on Line 50). One approach is to run a loop for each element 
    in the vector, calculate the distance and add all results but here, we want to use the power of vectorization to calculate the
    euclidean distance WITHOUT using any loops. Do not use in-built Euclidean distance functions.  

    '''
    distance = 0.0
    for i in range(len(x1) - 1):
        distance += np.square(x1[i] - x2[i])
    return np.sqrt(distance)


class KNearestNeighbors:

    def __init__(self, K=3):
        self.K = K

    def fit(self, x_train, y_train):
        self.X_train = x_train
        self.Y_train = y_train

    def predict(self, X_test):

        # list to store all our predictions
        predictions = []

        # loop over all observations in the test set
        for i in range(len(X_test)):

            # calculate the distance between the test point and all other points in the training set
            dist = np.array([euc_dist(X_test[i], x_t) for x_t in self.X_train])

            # sort the distances and return the indices of K neighbors store the result in 'dist_sorted'
            dist_sorted = dist.argsort()[:self.K]

            # get the neighbors
            neigh_count = {}

            # for each neighbor find the class
            for idx in dist_sorted:
                if self.Y_train[idx] in neigh_count:
                    neigh_count[self.Y_train[idx]] += 1
                else:
                    neigh_count[self.Y_train[idx]] = 1

            sorted_neigh_count = sorted(neigh_count.items(), key=operator.itemgetter(1), reverse=True)

            # append the class label to the list
            predictions.append(sorted_neigh_count[0][0])
        return predictions


def main():
    # CODING REQUIRED ###

    np.random.seed(700)  # DO NOT CHANGE

    # Load the iris dataset from sklearn.datasets
    iris = load_iris()

    # Store the features in variable 'X'
    X = iris.data

    # Store the targets in variable 'y'
    y = iris.target

    # Use the train_test_split function that you learnt in previous exercises to perform the split of the iris dataset.
    # Split 80% of the data into training and 20% to testing

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNearestNeighbors()
    knn.fit(X_train, y_train)

    # Use the class provided 'KNearestNeighbors' to fit and predict on the splits created. More specifically create an object
    # of the class, then, call the fit and predict functions by passing approrpriate parameters

    # Some code here

    predictions = knn.predict(X_test)

    print('Accuracy:', accuracy_score(y_test, predictions))


if __name__ == "__main__":
    main()
