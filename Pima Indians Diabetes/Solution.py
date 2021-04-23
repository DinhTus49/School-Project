import numpy as np
import pandas as pd
import sys
import utils
from scipy import optimize
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

np.set_printoptions(threshold=sys.maxsize)


def randomInitialWeights(L_in, L_out, epsilon=0.12):
    # random initialize weights of a layer
    '''
     :param L_in: number of incomming connections
     :param L_out: number of outgoing connections
     :param epsilon: self explanatory
     :return: W (array like): Initialize W randomly so that we break the symmetry while training
    the neural network. Note that the first column of W corresponds
    to the parameters for the bias unit.
     '''
    W = np.zeros((L_out, L_in + 1))
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon - epsilon
    return W


def nnCostFunction(nn_params,
                   input_neurons,
                   hidden_neurons,
                   num_labels,
                   X, y, lambda_=0):
    theta1 = np.reshape(nn_params[:hidden_neurons * (input_neurons + 1)],
                        (hidden_neurons, (input_neurons + 1)))  # shape: (5, 9)
    theta2 = np.reshape(nn_params[hidden_neurons * (input_neurons + 1):],
                        (num_labels, (hidden_neurons + 1)))  # shape: (1, 6)
    J = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    m = y.size
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # feed forward
    a1 = X  # shape: (m, 9)
    z2 = np.dot(a1, theta1.T)  # shape: (m, 5)
    a2 = utils.sigmoid(z2)
    a2 = np.concatenate((np.ones((a2.shape[0], 1)), a2), axis=1)  # shape: (m, 6)
    z3 = np.dot(a2, theta2.T)
    a3 = utils.sigmoid(z3)  # shape: (m, 1)

    # compute cost
    first_term = np.dot(y, np.log(a3))
    second_term = np.dot((1 - y), np.log(1 - a3))
    J = -1 / m * (first_term + second_term)

    # reg term
    reg_term = 0
    reg_term += sum(theta1[:, 1:].ravel() ** 2)
    reg_term += sum(theta2[:, 1:].ravel() ** 2)
    J += lambda_ / (2 * m) * reg_term

    # backprop
    capital_delta_1 = np.zeros(theta1.shape)  # shape: (5, 9)
    capital_delta_2 = np.zeros(theta2.shape)  # shape: (1, 6)
    for i in range(m):
        delta_3 = a3[i] - y[i]  # shape: (1, 1)
        delta_2 = np.multiply(np.dot(theta2[:, 1:].T, delta_3), utils.sigmoidGradient(z2[i]))
        capital_delta_1 += np.outer(delta_2, a1[i])
        capital_delta_2 += np.outer(delta_3, a2[i])
    theta1_grad = 1 / m * capital_delta_1
    theta2_grad = 1 / m * capital_delta_2

    # regularized
    theta1_grad[:, 1:] += lambda_ / m * theta1[:, 1:]
    theta2_grad[:, 1:] += lambda_ / m * theta2[:, 1:]
    grad = np.concatenate((theta1_grad.ravel(), theta2_grad.ravel()))
    return J, grad


def trainNeuralNetwork(X, y, lambda_, input_neurons,
                       output_neurons, hidden_layer_neurons,
                       initial_nn_params):
    options = {'maxiter': 100}
    costFunction = lambda p: nnCostFunction(p, input_neurons,
                                            hidden_layer_neurons,
                                            output_neurons, X, y, lambda_)
    res = optimize.minimize(costFunction,
                            initial_nn_params,
                            jac=True,
                            method='TNC',
                            options=options)
    all_theta = res.x
    return all_theta


def predict(X, nn_params, input_neurons, hidden_layer_neurons, output_neurons):
    if X.ndim == 1:
        X = X[None]
    m = X.shape[0]
    p = []
    theta1 = np.reshape(nn_params[:(hidden_layer_neurons * (input_neurons + 1))],
                        (hidden_layer_neurons, input_neurons + 1))
    theta2 = np.reshape(nn_params[(hidden_layer_neurons * (input_neurons + 1)):],
                        (output_neurons, hidden_layer_neurons + 1))
    h1 = utils.sigmoid(np.dot(np.concatenate([np.ones((m, 1)), X], axis=1), theta1.T))
    h2 = utils.sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), theta2.T))
    for i in h2:
        if i > 0.5:
            p.append(1)
        else:
            p.append(0)
    return p


if __name__ == '__main__':
    # add title and create csv file for pandas
    with open("pima-indians-diabetes.csv") as f:
        data = f.read()
    title = "Times pregnant,Plasma glucose concentration,Diastolic blood pressure,Triceps skin fold thickness," \
            "2-Hour serum insulin,Body mass index,Diabetes pedigree function,Age,Class\n"
    new_data = title + data
    f = open("Pima_Indians_Diabetes.csv", 'w')
    f.write(new_data)
    f.close()

    # load data to X, y
    data = pd.read_csv("Pima_Indians_Diabetes.csv", delimiter=',')
    print(data.columns.tolist())  # xem xem tại sao thỉnh thoảng lấy data bị lỗi do string k đúng
    y = np.array(data["Class"], dtype=np.float)
    features = ["Times pregnant", "Plasma glucose concentration", "Diastolic blood pressure",
                "Triceps skin fold thickness",
                "2-Hour serum insulin", "Body mass index", "Diabetes pedigree function", "Age"]
    X = np.array(data[features], dtype=np.float)
    m, n = X.shape

    # train, validation and test set
    X_train = X[:int(m * 0.6), :]
    X_val = X[int(m * 0.6):int(m * 0.8), :]
    X_test = X[int(m * 0.8):, :]

    y_train = y[:int(m * 0.6)]
    y_val = y[int(m * 0.6):int(m * 0.8)]
    y_test = y[int(m * 0.8):]

    X_train_tf = np.concatenate((X_train, X_val), axis=0)
    y_train_tf = np.concatenate((y_train, y_val), axis=0)

    # create neural network model
    INPUT_NEURONS = n
    OUTPUT_NEURON = 1
    HIDDEN_LAYER_NEURONS = 5

    # lambda_ = 3
    # utils.checkNNGradients(nnCostFunction, lambda_)
    theta1 = randomInitialWeights(INPUT_NEURONS, HIDDEN_LAYER_NEURONS, epsilon=0.68)
    theta2 = randomInitialWeights(HIDDEN_LAYER_NEURONS, OUTPUT_NEURON, epsilon=1)
    nn_params = np.concatenate([theta1.ravel(), theta2.ravel()], axis=0)
    J, grad = nnCostFunction(nn_params, INPUT_NEURONS, HIDDEN_LAYER_NEURONS, OUTPUT_NEURON,
                             X_train, y_train, lambda_=0)
    print("Initial cost: " + str(J))
    #print(predict(X_train[:6], nn_params, INPUT_NEURONS, HIDDEN_LAYER_NEURONS, OUTPUT_NEURON))
    #print("Real:" + str(y_train[:6]))
    lambda_ = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 50]
    all_params = np.zeros((len(lambda_), HIDDEN_LAYER_NEURONS * (INPUT_NEURONS + 1) +
                           OUTPUT_NEURON * (HIDDEN_LAYER_NEURONS + 1)))
    best_cost = sys.float_info.max
    chosen_params = np.zeros((HIDDEN_LAYER_NEURONS * (INPUT_NEURONS + 1) +
                              OUTPUT_NEURON * (HIDDEN_LAYER_NEURONS + 1)))
    all_J_train = []
    for i in range(len(lambda_)):
        temp_params = trainNeuralNetwork(X_train, y_train, lambda_=lambda_[i],
                                         input_neurons=INPUT_NEURONS,
                                         output_neurons=OUTPUT_NEURON,
                                         hidden_layer_neurons=HIDDEN_LAYER_NEURONS,
                                         initial_nn_params=nn_params)
        all_params[i] = temp_params
        J_train, _ = nnCostFunction(temp_params, INPUT_NEURONS, HIDDEN_LAYER_NEURONS, OUTPUT_NEURON,
                                   X_test, y_test, lambda_=0)
        all_J_train.append(J_train)
    print("\n \n")
    all_J_val = []
    for i in range(all_params.shape[0]):
        J_val, _ = nnCostFunction(all_params[i], INPUT_NEURONS, HIDDEN_LAYER_NEURONS, OUTPUT_NEURON,
                                  X_val, y_val, lambda_=0)
        print("J val: " + str(J_val) + " with params:" + str(all_params[i]))
        all_J_val.append(J_val)
        print("\n")
        if J_val < best_cost:
            best_cost = J_val
            chosen_params = all_params[i]
    print("\n \n")
    print("Chosen params: " + str(chosen_params))
    print("Best cost: " + str(best_cost))
    right = 0
    predictions = predict(X_test, chosen_params, INPUT_NEURONS, HIDDEN_LAYER_NEURONS, OUTPUT_NEURON)
    for i in range(len(predictions)):
        if predictions[i] == y_test[i]:
            right += 1
    print("Accuracy: " + str(right/y_test.size))
    '''plt.plot(all_J_val, 'r')
    plt.plot(all_J_train, 'b')
    plt.show()'''

    # tensorflow model
    '''model = keras.models.Sequential([
        keras.layers.Dense(INPUT_NEURONS),
        keras.layers.Dense(HIDDEN_LAYER_NEURONS, activation="sigmoid"),
        keras.layers.Dense(OUTPUT_NEURON, activation="sigmoid")
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_tf, y_train_tf, epochs=100, verbose=2, validation_split=0.2)
    tf_score = 0'''
    '''for i in range(y_test.size):
        test_X = np.reshape(X_test[i], (1, 8))
        print(test_X.shape)
        #pred = model.predict(X_test[i])
        #print(pred)
        break'''

