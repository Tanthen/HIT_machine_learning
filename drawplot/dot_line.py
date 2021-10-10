from matplotlib import pyplot as plt
import numpy as np


def binary_func(x, A=None):
    if A is None:
        A = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    result = 0
    for i in range(len(A)):
        result += A[i] * (x ** i)
    return result


def dot_line(w, sample_x, sample_y, fig, row, col, number):
    # fig = plt.figure()
    X = np.linspace(-1, 1)
    Y = binary_func(X, w)
    ax = fig.add_subplot(row, col, number)
    ax.plot(X, Y)

    plt.scatter(sample_x, sample_y, color='red', marker='+')


def acc_plot(x, y1, y2, fig, row, col, number):
    ax = fig.add_subplot(row, col, number)
    ax.plot(x, y1)
    plt.plot(x, y1, color='orange')
    plt.plot(x, y2, color='green')
