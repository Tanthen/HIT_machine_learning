import numpy as np
import math
from matplotlib import pyplot as plt
import os
debug = True


def loss(w, x, y, m):
    result = 0
    for i in range(m):
        result -= y*np.log(my_sigmoid(w,x)) + (1-y)*np.log(my_sigmoid(-w,x))
    return result


def logistic_model_grad(x, y, t_x, t_y, epoch=10000, a=0.03, threshold=1e-6, reg_lambda =1e-8):
    assert x.shape[0] == y.shape[0]
    m = x.shape[0]
    t_m = t_x.shape[0]
    ones = np.ones((m, 1))
    t_ones = np.ones((t_m, 1))
    # x = np.c_[ones, x] / m
    x = np.c_[ones, x]
    t_x = np.c_[t_ones, t_x]
    n = x.shape[1]
    w = np.random.rand(n)
    e = 0
    nor = 0.1
    times=[]
    acc_train=[]
    acc_test=[]
    while nor > threshold and e < epoch:
        e += 1
        g = np.zeros(n)
        # if e > epoch/3:
        #     a = 0.005
        # if e > epoch*2/3:
        #     a = 1e-5
        for i in range(m):
            g -= (y[i]*x[i] - my_sigmoid(w, x[i])*x[i] + reg_lambda*w) / m
        delta_w = a * g
        w -= delta_w
        old_nor = nor
        nor = np.linalg.norm(g)
        #print("loss", loss(w,x[i],y[i],m))
        #print("a", a)
        acc = 0
        times.append(e)
        for i in range(m):
            if my_sigmoid(w, x[i]) > 0.5 and y[i] == 1:
                acc += 1
            elif my_sigmoid(w, x[i]) < 0.5 and y[i] == 0:
                acc += 1
        acc_train.append(acc/m)
        acc = 0
        for i in range(t_m):
            if my_sigmoid(w, t_x[i]) > 0.5 and y[i] == 1:
                acc += 1
            elif my_sigmoid(w, t_x[i]) < 0.5 and y[i] == 0:
                acc += 1
        acc_test.append(acc / t_m)
        #print("nor:", nor)
        #print("e:", e)
    return w, times, acc_train, acc_test



#
# def logistic_model_conj(x, y, reg_lambda=1e-4):
#     assert x.shape[0] == y.shape[0]
#     m = x.shape[0]
#     ones = np.ones((m, 1))
#     print(ones)
#     print(x)
#     x = np.c_[ones, x]
#     n = x.shape[1]
#     w = np.random.rand(n)
#     r = np.zeros(n)
#     for i in range(m):
#         r += y[i] * x[i] - (math.exp(np.vdot(w, x[i])) / (1 + math.exp(np.vdot(w, x[i])))) * x[i] + reg_lambda * w
#     p = -r
#     for e in range(n):
#         a = np.vdot(r, r)

#
# def my_sigmoid(w, x):
#     if np.vdot(w, x) < 40:
#         return 1/


def my_sigmoid(w, x):
    if np.vdot(w, x) < 0:
        return np.exp(np.vdot(w, x))/(np.exp(np.vdot(w, x)) + 1)
    else:
        return 1 / (1 + np.exp(-np.vdot(w, x)))


def logistic_model_newton(x, y, t_x, t_y, threshold=1e-8 , reg_lambda=1e-8, epoch= 100):
    assert x.shape[0] == y.shape[0]
    m = x.shape[0]
    t_m = t_x.shape[0]
    ones = np.ones((m, 1))
    t_ones = np.ones((t_m, 1))
    #x = np.c_[ones, x] / m
    x = np.c_[ones, x]/(m+t_m)
    t_x = np.c_[t_ones, t_x] / (m+t_m)
    n = x.shape[1]
    w = np.random.rand(n)
    w = w.reshape(n, 1)
    nor = 0.1
    e = 0
    acc_train=[]
    acc_test=[]
    times=[]

    times.append(e)
    acc = 0
    for i in range(m):
        if my_sigmoid(w, x[i]) > 0.5 and y[i] == 1:
            acc += 1
        elif my_sigmoid(w, x[i]) < 0.5 and y[i] == 0:
            acc += 1
    acc_train.append(acc / m)
    acc = 0
    for i in range(t_m):
        if my_sigmoid(w, t_x[i]) > 0.5 and t_y[i] == 1:
            acc += 1
        elif my_sigmoid(w, t_x[i]) < 0.5 and t_y[i] == 0:
            acc += 1
    acc_test.append(acc / t_m)

    while nor > threshold and e <= epoch:
        e += 1
        H = np.zeros((n, n))
        g = np.zeros((n, 1))
        for i in range(m):
            H += my_sigmoid(w, x[i])*(1 - my_sigmoid(w, x[i])) * np.matmul(x[i].reshape(n, 1), x[i].reshape(1, n))
            g += -(y[i]*x[i] - my_sigmoid(w, x[i]) * x[i]).reshape(n, 1)
        g += reg_lambda * w
        H += reg_lambda * np.identity(n)
        delta_w = np.matmul(np.linalg.pinv(H), -g)
        nor = np.linalg.norm(delta_w)
        #     break
        w = w + delta_w

        #print(delta_w)
        times.append(e)
        acc = 0
        for i in range(m):
            if my_sigmoid(w, x[i]) > 0.5 and y[i] == 1:
                acc += 1
            elif my_sigmoid(w, x[i]) < 0.5 and y[i] == 0:
                acc += 1
        acc_train.append(acc/m)
        acc = 0
        for i in range(t_m):
            if my_sigmoid(w, t_x[i]) > 0.5 and t_y[i] == 1:
                acc += 1
            elif my_sigmoid(w, t_x[i]) < 0.5 and t_y[i] == 0:
                acc += 1
        acc_test.append(acc/t_m)
    return w, times, acc_train, acc_test


def generate_data(N, naive=True ,posRate=0.4):
    posNumber = np.ceil(N * posRate).astype(np.int32)
    sigma = [0.3, 0.4]
    cov12 = 0.2
    pos_mean = [1, 1]
    neg_mean = [0.5, 0.5]
    x = np.zeros((N, 2))
    y = np.zeros(N).astype(np.int32)
    if naive:
        x[:posNumber, :] = np.random.multivariate_normal(pos_mean, [[sigma[0], 0], [0, sigma[1]]],
                                                         size=posNumber)
        x[posNumber:, :] = np.random.multivariate_normal(neg_mean, [[sigma[0], 0], [0, sigma[1]]],
                                                         size=N - posNumber)
        y[:posNumber] = 1
        y[posNumber:] = 0
    else:
        x[:posNumber, :] = np.random.multivariate_normal(pos_mean, [[sigma[0], cov12], [cov12, sigma[1]]],
                                                         size=posNumber)
        x[posNumber:, :] = np.random.multivariate_normal(neg_mean, [[sigma[0], cov12], [cov12, sigma[1]]],
                                                         size=N - posNumber)
        y[:posNumber] = 1
        y[posNumber:] = 0
    return x, y


if __name__ == '__main__':
    pos_num = 30
    neg_num = 20
    x, y = generate_data(pos_num+neg_num, naive=False, posRate=pos_num/(pos_num + neg_num))
    w1,_,_,_ = logistic_model_newton(x, y, x, y)
    w2,_,_,_ = logistic_model_grad(x, y, x,y,10000)
    fig = plt.figure()
    X = np.linspace(-1, 2)
    print("w.shape", w1.shape)
    print("X.shape", X.shape)
    w1 = w1.squeeze()
    w2 = w2.squeeze()
    Y = (w1[0] + np.dot(w1[1], X))/(-w1[2])
    Y2 =(w2[0] + np.dot(w2[1], X))/(-w2[2])
    ax1 = fig.add_subplot(211)
    ax1.plot(X, Y, color="blue")
    ax1.scatter(x[:pos_num, 0], x[:pos_num, 1], color="red", marker="+")
    ax1.scatter(x[pos_num:, 0], x[pos_num:, 1], color="green", marker="_")

    ax2 = fig.add_subplot(212)
    ax2.plot(X, Y2, color="orange")
    ax2.scatter(x[:pos_num, 0], x[:pos_num, 1], color="red", marker="+")
    ax2.scatter(x[pos_num:, 0], x[pos_num:, 1], color="green", marker="_")

    plt.show()
