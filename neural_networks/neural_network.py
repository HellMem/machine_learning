import pandas as pd
import numpy as np
import copy

INPUT_LAYER_SIZE = 4
HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = 3
OUTPUT_LAYER_SIZE = 3


# Inicia los pesos de cada capa de manera aleatoria
def init_weights():
    weights = []

    weights.append(np.random.randn(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE) * np.sqrt(2.0 / INPUT_LAYER_SIZE))

    for i in range(HIDDEN_LAYERS - 1):
        weights.append(np.random.randn(HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE) * np.sqrt(2.0 / INPUT_LAYER_SIZE))

    weights.append(np.random.randn(HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE) * np.sqrt(2.0 / HIDDEN_LAYER_SIZE))

    return weights


# Inicia los biases de cada capa de manera aleatoria
def init_bias():
    biases = []

    for i in range(HIDDEN_LAYERS):
        biases.append(np.full((1, HIDDEN_LAYER_SIZE), 0.1))

    biases.append(np.full((1, OUTPUT_LAYER_SIZE), 0.1))
    return biases


# Función de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Calcula la derivada de la activación
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


# Utilizando la función Cross-Entropy, determina el costo del modelo recibido para el data set
def cost_function(X, Y, weights, biases):
    cost = 0

    for x, y in zip(X, Y):
        predictions = feed_forward(x, weights, biases)

        part1_cost = y * np.log(predictions)

        part2_cost = (1 - y) * np.log(1 - predictions)

        cost += (part1_cost.sum() + part2_cost.sum())

    return -cost / len(X)


# Esta función obtiene una predicción para una entrada por medio de unos pesos y biases
def feed_forward(X, weights, biases):
    H = X
    for i in range(HIDDEN_LAYERS + 1):
        weight = weights[i]
        bias = biases[i]
        Zi = np.dot(H, weight) + bias
        H = sigmoid(Zi)

    return H


# Esta función ejecuta el algoritmo de aprendizaje back propagation
def back_prop(X, Y, weights, biases):
    error_weights = [np.zeros(weight.shape) for weight in weights]
    error_biases = [np.zeros(bias.shape) for bias in biases]

    for x, y in zip(X, Y):
        delta_errors_weights = [np.zeros(weight.shape) for weight in weights]
        delta_errors_biases = [np.zeros(bias.shape) for bias in biases]

        activation = x
        activations = [x]

        zs = []

        for b, w in zip(biases, weights):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        delta_errors_biases[-1] = delta
        delta_errors_weights[-1] = np.dot(activations[-2].transpose(), delta)

        #maybe change this loop, use foreach instead.
        #It might be clearear on how it is going backwards
        #use zip with all parameters

        for l in range(2, HIDDEN_LAYERS + 1):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(delta, weights[-l + 1].transpose()) * sp
            delta_errors_biases[-l] = delta
            delta_errors_weights[-l] = np.dot(activations[-l - 1].transpose(), delta)

        error_weights = [ew + dew for ew, dew in zip(error_weights, delta_errors_weights)]
        error_biases = [eb + deb for eb, deb in zip(error_biases, delta_errors_biases)]

    return error_weights, error_biases


# Se ejecuta el gradiente descendiente para determinar el módelo óptimo
def train(X, y, weights, biases, alpha, iters):
    training_set_size = len(X)
    history = {'cost': [], 'iter': []}

    cost = 100000
    i = 0
    for i in range(iters):
    #while cost > 0.01:
        gradW, gradB = back_prop(X, y, weights, biases)
        weights_copy = copy.deepcopy(weights)
        weights = []
        for w, g in zip(weights_copy, gradW):
            w = np.add(w, - (g * alpha / training_set_size))
            weights.append(w)

        biases_copy = copy.deepcopy(biases)
        biases = []
        for b, g in zip(biases_copy, gradB):
            b = np.add(b, - (g * alpha / training_set_size))
            biases.append(b)

        cost = cost_function(X, y, weights, biases)
        history['cost'].append(cost)
        history['iter'].append(i)

        if i % 10 == 0:
            print("iter: " + str(i) + " cost: " + str(cost))

    return [weights, biases, history]


def cost_derivative(output_activations, y):
    return output_activations - y


if __name__ == "__main__":
    weights = init_weights()
    biases = init_bias()

    iris = pd.read_csv("iris.csv")

    values = iris.sample(frac=1).values

    test_values = values[:45]
    training_values = values[0:105]

    X = copy.deepcopy(training_values)
    y = copy.deepcopy(training_values)
    X = np.delete(X, 4, 1)
    X = np.delete(X, 4, 1)
    X = np.delete(X, 4, 1)

    y = np.delete(y, 0, 1)
    y = np.delete(y, 0, 1)
    y = np.delete(y, 0, 1)
    y = np.delete(y, 0, 1)
    [weights, biases, cost_history] = train(X, y, weights, biases, 0.3, 20000)

    y_test_values = copy.deepcopy(test_values)

    test_values = np.delete(test_values, 4, 1)
    test_values = np.delete(test_values, 4, 1)
    test_values = np.delete(test_values, 4, 1)

    y_test_values = np.delete(y_test_values, 0, 1)
    y_test_values = np.delete(y_test_values, 0, 1)
    y_test_values = np.delete(y_test_values, 0, 1)
    y_test_values = np.delete(y_test_values, 0, 1)

    correct_predictions = 0

    for x, y in zip(X, y):
        pred = feed_forward(x, weights, biases)
        pred_index = np.argmax(pred)
        if y[pred_index] == 1:
            correct_predictions += 1

    print(str(correct_predictions) + " correct predictions from " + str(len(training_values)) + " training data")

    correct_predictions = 0

    for x, y in zip(test_values, y_test_values):
        pred = feed_forward(x, weights, biases)
        pred_index = np.argmax(pred)
        if y[pred_index] == 1:
            correct_predictions += 1

    print(str(correct_predictions) + " correct predictions from " + str(len(test_values)) + " tests")
