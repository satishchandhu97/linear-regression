from __future__ import division
import pandas as pd
import matplotlib.pyplot as plt
import random as rdm


def mean(value):
    return sum(value)/len(value)


def variance(value, mean):
    return sum([(x-mean)**2 for x in value])


def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += ((x[i]-mean_x)*(y[i]-mean_y))
    return covar


def coefficient(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - (b1 * x_mean)
    return [b0,b1]


def simple_linear_regression(train, test):
    predictions = []
    b0, b1 = coefficient(train)
    for row in test:
        yhat = b0 + (b1 * row[0])
        predictions.append(yhat)
    print('b0 :', b0, 'b1: ', b1, '* x')
    return predictions


df = pd.read_csv('imports-85.data.csv')
#  x = Automobile Engine Size
x = df.iloc[:, 16]
# y = Prize
y = df.iloc[:, 25]


data_test = [[x[i], y[i]] for i in range(len(df))]


data_train = []
length_train = (int(len(data_test)*0.70))
for i in range(length_train):
    rand_num = rdm.randrange(len(data_test))
    data_train.append(data_test.pop(rand_num))

print(len(data_test), len(data_train))


y_predict = simple_linear_regression(data_train, data_test)

x_filter = [row[0] for row in data_test]
y_filter = [row[1] for row in data_test]

plt.scatter(x_filter, y_filter)
plt.plot(x_filter, y_predict, color='blue', linewidth=1)
plt.show()