import numpy as np


def predict(features, weights):
    '''
    Returns 1D array of probabilities
    that the class label == 1
    '''
    z = np.dot(features, weights)
    return sigmoid(z)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# TODO change this function to corresponding cost function (Mean Square Error)
def cost_function(features, labels, weights):
    '''
    Using Mean Absolute Error

    Features:(Data Size, Number of features)
    Labels: (Data Size, 1)
    Weights:(Number of features,1)
    Returns 1D matrix of predictions
    Cost = (labels*log(predictions) + (1-labels)*log(1-predictions) ) / len(labels)
    '''
    observations = len(labels)

    predictions = predict(features, weights)

    # Take the error when label=1
    class1_cost = labels * np.log(predictions)

    # Take the error when label=0
    class2_cost = (1 - labels) * np.log(1 - predictions)

    # Take the sum of both costs
    cost = class1_cost + class2_cost

    # Take the average cost
    cost = cost.sum() / observations

    return -cost


# TODO change this function to corresponding gradient descent (Mean Square Error)
def update_weights(features, labels, weights, lr):
    '''
    Vectorized Gradient Descent

    Features:(Data Size, Number of features)
    Labels: (Data Size, 1)
    Weights:(Number of features, 1)
    '''
    N = len(features)

    # 1 - Get Predictions
    predictions = predict(features, weights)

    # 2 Transpose features from (200, 3) to (3, 200)
    # So we can multiply w the (200,1)  cost matrix.
    # Returns a (3,1) matrix holding 3 partial derivatives --
    # one for each feature -- representing the aggregate
    # slope of the cost function across all observations
    gradient = np.dot(features.T, predictions - labels)

    # 3 Take the average cost derivative for each feature
    gradient /= N

    # 4 - Multiply the gradient by our learning rate
    gradient *= lr

    # 5 - Subtract from our weights to minimize cost
    weights -= gradient

    return weights


def decision_boundary(prob):
    return 1 if prob >= .5 else 0


def train(features, labels, weights, lr, iters):
    cost_history = []

    for i in range(iters):
        weights = update_weights(features, labels, weights, lr)

        # Calculate error for auditing purposes
        cost = cost_function(features, labels, weights)
        cost_history.append(cost)

        # Log Progress
        if i % 10 == 0:
            print("iter: " + str(i) + " cost: " + str(cost))

    return weights, cost_history


if __name__ == "__main__":
    print('Linear Regression')
    f = 0
