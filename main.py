# This is a sample Python script.

import matplotlib.pyplot as plt
from model import polyfit
from data_generate import data_generate
import numpy as np
import math
from drawplot import dot_line


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def binary_func(x, A=None):
    if A is None:
        A = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    result = 0
    for i in range(len(A)):
        result += A[i] * (x ** i)
    return result


def sin2px(x):
    return math.sin(2 * math.pi * x)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # mod = polyfit.Poly_Fit()
    # x, y = data_generate.generate_data(function=sin2px, n=25)
    # w, loss = mod.parse_result(x, y, order=20, reg_lambda=0.0001)
    # # w, loss = mod.conj_grad(x, y, order=50, reg_lambda=0.0001 ,threshold=1e-15)
    # print(loss)
    # fig = plt.figure(figsize=(1920, 1080), dpi=100)
    # dot_line.dot_line(w, x, y, fig, 5, 5, 9)
    # dot_line.dot_line(w, x, y, fig, 5, 5, 10)
    # dot_line.dot_line(w, x, y, fig, 5, 5, 12)
    # plt.show()

    mod = polyfit.Poly_Fit()
    data_scale = [5, 10, 25, 50, 100]
    epoch = [30000]
    lr = [0.5, 0.3, 0.1, 0.07, 0.03, 0.01, 0.001]
    reg_l = [1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    order = [6, 9, 20, 50]
    position = 1
    fig = plt.figure(figsize=(1920, 1080), dpi=100)
    row = 5
    col = 9

    for s in data_scale:
        x, y = data_generate.generate_data(function=sin2px, n=s)
        w, _ = mod.parse_result(x, y, order=len(x), reg_lambda=1e-7)
        print("parse result, position =", end=" ")
        print(position)
        dot_line.dot_line(w, x, y, fig, row, col, position)
        position += 1
        for o in order:
            w, t, y1, y2 = mod.conj_grad(x, y, o, 1e-7)
            print("parse result, position =", end=" ")
            print(position)
            dot_line.dot_line(w,x,y,fig,row,col,position)
            position += 1
            dot_line.acc_plot(t, y1, y2, fig, row, col, position)
            position += 1

    plt.show()





