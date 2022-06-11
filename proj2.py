#!/usr/bin/env python
# coding: utf-8

from copy import deepcopy
import math
import pandas as pd
from sklearn.model_selection import train_test_split


def euclidian_distance(v1, v2, columns):
    summation = 0
    for col in columns:
        summation += math.pow(v1[col] - v2[col], 2)
    return math.sqrt(summation)

def manhattan_distance(v1, v2, columns):
    summation = 0
    for col in columns:
        summation += abs(v1[col] - v2[col])
    return summation

def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())

def calc_accuracy(Y_pred):
    global Y_test
    score = 0.0
    length = len(Y_test)
    for i in range(length):
        if Y_test[i] == Y_pred[i]:
            score = score + 1
    return score / length

def predict(current_features):
    global X_train, X_test, Y_train
    global num_of_features
    Y_pred = []
    for test_row in X_test:
        min_dist = float("inf")
        prediction = 0
        for index, train_row in enumerate(X_train):
            distance = euclidian_distance(train_row, test_row, current_features)
            if distance < min_dist:
                min_dist = distance
                prediction = Y_train[index]
        Y_pred.append(prediction)
    return Y_pred

def get_next_best(best_features):
    global num_of_features
    accuracy = 0.0
    features = []
    for feature in range(num_of_features):
        if feature not in best_features:
            current_features = deepcopy(best_features)
            current_features.append (feature)
            y_pred = predict(current_features)
            accuracy_ = calc_accuracy(y_pred)
            if accuracy_ > accuracy:
                features = current_features
                accuracy = accuracy_
    return features, accuracy

def forward_selection ():
    global num_of_features
    features = []
    for step in range(num_of_features):
        features, accuracy = get_next_best(features)
        print(f"step: {step}, score: {accuracy:.3f}, features: {features}")


def eliminate_worst(last_features):
    global num_of_features
    accuracy = 0.0
    features = []
    for feature in range(num_of_features):
        if feature in last_features:
            current_features = deepcopy(last_features)
            current_features.remove (feature)
            y_pred = predict(current_features)
            accuracy_ = calc_accuracy(y_pred)
            if accuracy_ > accuracy:
                features = current_features
                accuracy = accuracy_
    return features, accuracy

def backward_elimination ():
    global num_of_features
    features = [x for x in range(num_of_features)]
    y_pred = predict(features)
    accuracy = calc_accuracy(y_pred)
    print(f"step: 0, score: {accuracy:.3f}, features: {features}")
    for step in range(num_of_features - 1):
        features, accuracy = eliminate_worst(features)
        print(f"step: {step + 1}, score: {accuracy:.3f}, features: {features}")

def run(input_file, algo, normalization):
    global num_of_features
    global X_train, X_test, Y_train, Y_test
    df = pd.read_csv(input_file, delim_whitespace=True, header=None)

    if normalization:
        for col in df.columns:
            if col != 0:
                df[col] = min_max_normalize(df[col])

    X = df.iloc[:, 1:].values.astype(float)
    Y = df.iloc[:, 0].values.astype(int)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=2, test_size=0.20)
    num_of_features = len(X[0])

    if algo == 1:
        print("Running forward_selection")
        forward_selection()
    else:
        print("Running backward_elimination")
        backward_elimination()


run(input_file='dataset/CS205_CalibrationData__1.txt', algo=1, normalization=True)
print("\n===================================================\n")
run(input_file='dataset/CS205_CalibrationData__1.txt', algo=2, normalization=True)

run(input_file='dataset/CS205_CalibrationData__2.txt', algo=1, normalization=True)
print("\n===================================================\n")
run(input_file='dataset/CS205_CalibrationData__2.txt', algo=2, normalization=True)

run(input_file='dataset/CS205_CalibrationData__3.txt', algo=1, normalization=True)
print("\n===================================================\n")
run(input_file='dataset/CS205_CalibrationData__3.txt', algo=2, normalization=True)


run(input_file='dataset/CS205_SP_2022_SMALLtestdata__24.txt', algo=1, normalization=True)
print("\n===================================================\n")
run(input_file='dataset/CS205_SP_2022_SMALLtestdata__24.txt', algo=2, normalization=True)


run(input_file='dataset/CS205_SP_2022_Largetestdata__5.txt', algo=1, normalization=True)
print("\n===================================================\n")
run(input_file='dataset/CS205_SP_2022_Largetestdata__5.txt', algo=2, normalization=True)


