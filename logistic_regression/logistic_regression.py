import numpy as np


def predict(features, weights):
    z = np.dot(features, weights)
    return sigmoid(z)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_function(features, labels, weights):
    observations = len(labels)

    predictions = predict(features, weights)

    class1_cost = labels * np.log(predictions)

    class2_cost = (1 - labels) * np.log(1 - predictions)

    cost = class1_cost + class2_cost

    cost = cost.sum() / observations

    return -cost


def update_weights(features, labels, weights, lr):
    N = len(features)

    predictions = predict(features, weights)
    gradient = np.dot(features.T, predictions - labels)

    gradient /= N
    gradient *= lr

    weights -= gradient

    return weights


def decision_boundary(prob):
    return 1 if prob >= .5 else 0


def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        if i % 10 == 0:
            print("iter: " + str(i) + " cost: " + str(cost))

    return weights, cost_history


if __name__ == "__main__":
    print('Logistic Regression')
