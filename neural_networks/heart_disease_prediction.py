import neural_network as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    taken_columns = 12
    heart_disease = pd.read_csv("heart.csv")

    values = heart_disease.sample(frac=1).values

    test_values = values[:92]
    training_values = values[0:212]

    y_train = training_values[:, 13]
    y_test = test_values[:, 13]

    x_train = training_values[:, 0:taken_columns]
    x_test = test_values[:, 0:taken_columns]

    nn.INPUT_LAYER_SIZE = taken_columns
    nn.HIDDEN_LAYERS = 1
    nn.HIDDEN_LAYER_SIZE = 15
    nn.OUTPUT_LAYER_SIZE = 1

    weights = nn.init_weights()
    biases = nn.init_bias()

    alpha = 3
    [weights, biases, cost_history] = nn.train(x_train, y_train, weights, biases, alpha, 50000)

    correct_predictions = 0

    for x, y in zip(x_train, y_train):
        pred = nn.feed_forward(x, weights, biases)
        if y == 1 and pred.sum() > 0.5:
            correct_predictions += 1
        elif y == 0 and pred.sum() <= 0.5:
            correct_predictions += 1

    print(str(correct_predictions) + " correct predictions from " + str(len(training_values)) + " train values")

    correct_predictions = 0

    for x, y in zip(x_test, y_test):
        pred = nn.feed_forward(x, weights, biases)
        if y == 1 and pred.sum() > 0.5:
            correct_predictions += 1
        elif y == 0 and pred.sum() <= 0.5:
            correct_predictions += 1

    print(str(correct_predictions) + " correct predictions from " + str(len(test_values)) + " tests")

    iter_history = cost_history.get('iter')
    cost_history = cost_history.get('cost')

    plt.plot(iter_history, cost_history, 'ro')
    plt.gca().set_title('learning rate: ' + str(alpha))
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.show()

