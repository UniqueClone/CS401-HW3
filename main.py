"""
Author: Ryan Lynch
CS401 Assignment 3
"""


import os
import numpy as np
import pandas as pd
from numpy import loadtxt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import glob


def plot_roc(labels, data, model):
    predictions = model.predict(data)
    fp, tp, _ = roc_curve(labels, predictions)
    plt.plot(100 * fp, 100 * tp)
    plt.xlabel("False Positive Rate [%]")
    plt.ylabel("True Positive Rate [%]")
    plt.show()


def kfoldCrossValidation(X, y, k=8):
    num_validation_samples = len(X) // k
    validation_scores = []
    for fold in range(k):
        valid_x = X[num_validation_samples * fold: num_validation_samples * (fold + 1)]
        valid_y = y[num_validation_samples * fold: num_validation_samples * (fold + 1)]
        train_x = np.concatenate((X[:num_validation_samples * fold], X[num_validation_samples * (fold + 1):]))
        train_y = np.concatenate((y[:num_validation_samples * fold], y[num_validation_samples * (fold + 1):]))
        cur_model = keras.models.load_model('ML_model')
        cur_model.fit(train_x, train_y, epochs=175)
        valid_x = np.asarray(valid_x)
        valid_y = np.asarray(valid_y)
        eval = cur_model.evaluate(valid_x, valid_y)
        validation_scores.append(eval)
    return validation_scores


hw3_files = sorted(glob('data/*'))

pd.concat((pd.read_csv(file) for file in hw3_files), axis='columns').head()


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# dataset = loadtxt('Files/train-io.txt', delimiter=' ')

# X = dataset[:, 0:10]
# y = dataset[:, 10]

# train_data_scaled = StandardScaler().fit_transform(X)

# The following commented-out lined perform k-fold cross validation
# results = kfoldCrossValidation(train_data_scaled, y, k=8)
# avg_acc = 0
# avg_loss = 0
# k = len(results)
# for i in results:
#     avg_loss += i[0] / k
#     avg_acc += i[1] / k
# print('AVG LOSS: ' + str(avg_loss))
# print('AVG ACCURACY: ' + str(avg_acc))
# print(avg_loss, avg_acc)

# model.fit(X, y, epochs=150, batch_size=32)
model = keras.models.load_model('ML_model')
_, accuracy = model.evaluate(train_data_scaled, y)
print('Accuracy: %.2f' % (accuracy * 100))

test_dataset = loadtxt('Files/test-i.txt', delimiter=' ')

X = test_dataset[:, 0:10]

testing_output = model.predict(X)

test_output_file = open('test-o.txt', 'w')
for prediction in testing_output:
    # Round the outputs to 0 or 1 and write to file
    test_output_file.write(str(round(float(prediction))) + '\n')

# Uncomment to plot ROC, output in Report.md
# plot_roc(y[-1000:], train_data_scaled[-1000:], model)

# data = pd.DataFrame(X, columns=range(1, 11))

print("\nDone")
