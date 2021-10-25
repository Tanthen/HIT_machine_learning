import numpy as np
from model import logistic
from data import *
import pandas as pd
from drawplot import dot_line
from matplotlib import pyplot as plt
if __name__ == '__main__':
    x = pd.read_csv("data/ionosphere.data", header=None)
    class_mapping = {'g': 1, 'b': 0}
    x[34] = x[34].map(class_mapping)
    x = x.sample(frac=1).reset_index(drop=True)
    train_data = x.iloc[:200, :]
    test_data = x.iloc[200:, :]

    x = train_data.iloc[:, :34].reset_index(drop=True)
    y = train_data.iloc[:, 34].reset_index(drop=True)
    t_x = test_data.iloc[:, :34].reset_index(drop=True)
    t_y = test_data.iloc[:, 34].reset_index(drop=True)
    reg = 10
    fig = plt.figure(figsize=(20, 10))
    for i in range(12):
        reg = reg/10

        w, times, acc_train, acc_test = logistic.logistic_model_newton(x, y, t_x, t_y, reg_lambda=reg)
        print(reg, ", ", acc_train[-1], ", ", acc_test[-1])
        ax = fig.add_subplot(3, 4, i+1)
        ax.plot(times, acc_train, color='orange')
        ax.plot(times, acc_test, color='red')
    plt.show()

    # w1, times1,acc_train1,acc_test1 = logistic.logistic_model_newton(x, y, t_x, t_y)
    # print(acc_train1[-1])
    # print(acc_test1[-1])
    # w2, _, acc_train2, _ = logistic.logistic_model_grad(x, y, t_x, t_y)
    # print(acc_train2[-1])
    # fig = plt.figure()
    # plt.ylim(0, 1)
    # plt.plot(times1, acc_train1, color='orange')
    # plt.plot(times1, acc_test1, color='red')
    # plt.show()

