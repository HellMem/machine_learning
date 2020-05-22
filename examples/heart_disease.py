import numpy as np
import pandas as pd
import logistic_regression.logistic_regression as log_reg
import random


def get_features_and_labels(data):
    # we separate the labels values from the matrix
    label_values = data[:, 13]
    features = np.delete(data, [13], axis=1)

    # we add a 1's column at the left (for the bias value)
    features = np.insert(features, 0, values=1, axis=1)

    return [features, label_values]


if __name__ == "__main__":
    heart = pd.read_csv("data/heart.csv")

    # learning rate
    lr = 0.01

    # iterations
    iterations = 1500

    # we get the features array
    train_data = (heart.values)

    [features, labels] = get_features_and_labels(train_data)

    # we create initial theta values
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01]
    weights = np.array(weights)

    [weights, cost_history] = log_reg.train(features, labels, weights, lr, iterations)
