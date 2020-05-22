import logistic_regression.logistic_regression as log_reg
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import math


def get_features_and_labels(data):
    # we separate the labels values from the matrix
    label_values = data[:, 4]
    features = np.delete(data, [4], axis=1)

    # we add a 1's column at the left (for the bias value)
    features = np.insert(features, 0, values=1, axis=1)

    return [features, label_values]


def get_iris_setosa_model(dataset, lr, iters):
    dataset['iris_class'].replace(to_replace=["Iris-setosa"], value=1, inplace=True)
    dataset['iris_class'].replace(to_replace=["Iris-versicolor"], value=0, inplace=True)
    dataset['iris_class'].replace(to_replace=["Iris-virginica"], value=0, inplace=True)

    # we create initial theta values
    weights = [1.1, 1.1, 1.1, 1.1, 1.1]
    weights = np.array(weights)

    [features, label_values] = get_features_and_labels(dataset.values)

    [weights, cost_history] = log_reg.train(features, label_values, weights, lr, iters)

    return weights


def get_iris_versicolor_model(dataset, lr, iters):
    dataset['iris_class'].replace(to_replace=["Iris-setosa"], value=0, inplace=True)
    dataset['iris_class'].replace(to_replace=["Iris-versicolor"], value=1, inplace=True)
    dataset['iris_class'].replace(to_replace=["Iris-virginica"], value=0, inplace=True)

    # we create initial theta values
    weights = [1.1, 1.1, 1.1, 1.1, 1.1]
    weights = np.array(weights)

    [features, label_values] = get_features_and_labels(dataset.values)

    [weights, cost_history] = log_reg.train(features, label_values, weights, lr, iters)

    return weights


def get_iris_virginica_model(data, lr, iters):
    data['iris_class'].replace(to_replace=["Iris-setosa"], value=0, inplace=True)
    data['iris_class'].replace(to_replace=["Iris-versicolor"], value=0, inplace=True)
    data['iris_class'].replace(to_replace=["Iris-virginica"], value=1, inplace=True)

    # we create initial theta values
    weights = [1.1, 1.1, 1.1, 1.1, 1.1]
    weights = np.array(weights)

    [features, label_values] = get_features_and_labels(data.values)

    [weights, cost_history] = log_reg.train(features, label_values, weights, lr, iters)

    return weights


def decision_boundary(prob):
    return 1 if prob >= .5 else 0


if __name__ == "__main__":
    iris = pd.read_csv("data/iris.csv")

    # learning rate
    lr = 0.01

    # iterations
    iterations = 1500

    # we get the features array
    train_data = iris.values

    iris_setosa_model = get_iris_setosa_model(copy.deepcopy(iris), lr, iterations)
    test_features = np.array([1.0, 5.1, 3.5, 1.4, 0.2])
    print(log_reg.predict(features=test_features, weights=iris_setosa_model))
    print('-' * 50)

    iris_versicolor_model = get_iris_versicolor_model(copy.deepcopy(iris), lr, iterations)

    print('-' * 50)
    iris_virginica_model = get_iris_virginica_model(copy.deepcopy(iris), lr, iterations)

    print(log_reg.predict(features=test_features, weights=iris_setosa_model))
    print(log_reg.predict(features=test_features, weights=iris_versicolor_model))
    print(log_reg.predict(features=test_features, weights=iris_virginica_model))
