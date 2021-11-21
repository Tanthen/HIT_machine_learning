import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import sys
import os
MAXN = sys.maxsize
raw = 4
col = 8


def euler_distance(x, u):
    assert len(x) == len(u)
    distance = 0
    for i in range(len(x)):
        distance += (x[i] - u[i])**2
    return distance


def k_means(X, k=3, drawable=True):
    m = X.shape[0]  # m：样本个数
    n = X.shape[1]  # n：样本维度
    assert m > k    # 确保样本个数大于k
    mu = X[:k]      # mu：中心点
    fig = plt.figure(figsize=(1920, 1080))
    number = 1
    change = True
    mu_list = dict()
    while change:
        mu_list = dict()
        for i in range(k):
            mu_list[i] = []
        # E-step
        for i in range(m):
            index = 0
            minn = MAXN
            for j in range(k):
                temp = euler_distance(X[i], mu[j])
                if temp < minn:
                    minn = temp
                    index = j
            mu_list[index].append(X[i])
        if drawable:
            k_means_drow(mu, mu_list, fig, number=number)
            number += 1
        # M-step
        change = False
        for i in range(k):
            if mu_list[i]:
                temp = np.zeros(n)
                for dot in mu_list[i]:
                    temp += dot
                temp = temp / len(mu_list[i])
                if euler_distance(temp, mu[i]) > 1e-14:
                    change = True
                    mu[i] = temp
        if drawable:
            k_means_drow(mu, mu_list, fig, number=number)
            number += 1
    if drawable:
        plt.show()
    return mu, mu_list


def k_means_drow(mu, mu_list, fig, number):
    color = ['red', 'green', 'blue', 'yellow']
    index = 0
    sp = fig.add_subplot(raw, col, number)
    for dots in mu_list:
        for x in mu_list[dots]:
            sp.scatter(x[0], x[1], color=color[index])
        index += 1
    print(mu)
    for dot in mu:
        sp.scatter(dot[0], dot[1], color='orange')


def generate_dot():
    mean1 = [12, 5]
    cov1 = [[2, 0.5], [0.5, 3]]
    data = np.random.multivariate_normal(mean1, cov1, 100)

    mean2 = [12, 12]
    cov2 = [[2, 0.1], [0.1, 1]]
    data = np.append(data, np.random.multivariate_normal(mean2, cov2, 100), 0)

    mean3 = [5, 10]
    cov3 = [[1, -0.7], [-0.7, 1]]
    data = np.append(data, np.random.multivariate_normal(mean3, cov3, 100), 0)

    return np.round(data, 4)


def gauss(x, cov, mu):
    assert x.shape[0] == mu.shape[0]
    assert x.shape[0] == cov.shape[0]
    assert cov.shape[0] == cov.shape[1]
    n = x.shape[0]
    # result = np.exp(-np.matmul(x-mu, np.matmul(cov, x-mu)) / 2) / ((2*np.pi)**(n/2) * np.linalg.norm(sqrtm(cov)))
    # print(x, mu, cov)
    cov_in = [cov[0][0], cov[1][1]]
    result = scipy.stats.multivariate_normal.pdf(x=x, mean=mu, cov=cov)
    return result


def gmm(X, k, drawable=True, cluster=None):
    assert k > 0
    m = X.shape[0]  # X的样本数
    n = X.shape[1]  # X的维度
    assert m > k

    mu = dict()
    sigma = dict()
    z = np.zeros(k)

    # initial
    for i in range(k):
        mu[i] = X[i]
        sigma[i] = np.identity(n)
        z[i] = 1/k
    r = np.zeros((m, k))
    # draw
    fig = plt.figure(figsize=(1920, 1080))
    number = 1
    if drawable:
        gmm_draw(mu, sigma, X, fig, number, start=True)
        number += 1
    last_mle = 0
    if not drawable:
        raw = 6
    else:
        raw = 4
    for e in range(raw*col - 1):
        # E-step
        point_list = dict()
        index_list = dict()
        for i in range(k):
            point_list[i] = []
            index_list[i] = []
        for i in range(m):
            for j in range(k):
                r[i][j] = z[j]*gauss(X[i], sigma[j], mu[j])
            point_list[np.argmax(r[i])].append(X[i])
            index_list[np.argmax(r[i])].append(i)
        mle = 0
        for j in range(k):
            for point in np.array(point_list[j]):
                 mle += z[j] * gauss(x=point, cov=sigma[j], mu=mu[j])

        print(np.abs(mle - last_mle))
        last_mle = mle

        r_sum = np.sum(r, axis=1)
        r = r / r_sum.reshape(m, 1)
        # M-step
        r_sum = np.sum(r, axis=0)  # r_sum:一行k个
        # --mu:
        for i in range(k):
            mu[i] = np.matmul(r[:, i].flatten(), X).flatten() / r_sum[i]  # 1*m X m*n == 1*n
        # -sigma:
        for j in range(k):
            sigma[j] = np.zeros((n, n))
            for i in range(m):
                sigma[j] += r[i][j]*np.matmul((X[i]-mu[j]).reshape(n, 1), (X[i]-mu[j]).reshape(1, n))
            sigma[j] = sigma[j] / r_sum[j]
        # --z:
        z = r_sum / m
        if drawable:
            gmm_draw(mu, sigma, point_list, fig, number, start=False)
            number += 1
        else:
            hit = np.zeros(6)
            alig = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
            for i in range(6):
                for my_cluster in index_list:
                    for index in index_list[my_cluster]:
                        # print("index:", index, "; cluster[index]:", cluster[index], " my_cluster:", my_cluster)
                        if cluster[index] == alig[i][my_cluster]:
                            hit[i] += 1
            print(hit)
            print("accuracy is ", max(hit) / m)
    return mu, sigma


def gmm_draw(mu, sigma, point, fig, number, start):
    sp = fig.add_subplot(raw, col, number)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    point_color = ['firebrick', 'gold', 'mediumblue', 'silver']
    color = ['darkred', 'goldenrod', 'darkblue', 'grey']
    index = 0
    if start:
        sp.scatter(point[:, 0], point[:, 1])
    else:
        for i in point:
            for dot in point[i]:
                sp.scatter(dot[0], dot[1], color=point_color[index])
            index += 1
    broaden = 4
    for i in range(len(mu)):
        s = sigma[i]
        u = mu[i]
        co = color[i]
        a = s[0][0]
        b = s[1][1]
        c = s[0][1]
        x = np.linspace(-np.sqrt(b / (a * b - c * c)), np.sqrt(b / (a * b - c * c)), 100)
        de = c * c * x * x - b * (a * x * x - 1)
        de = np.maximum(de, 0)
        y1 = (-c * x - np.sqrt(de)) / b
        y2 = (-c * x + np.sqrt(de)) / b
        sp.plot(broaden*x + u[0], broaden*y1 + u[1], color=co)
        sp.plot(broaden*x + u[0], broaden*y2 + u[1], color=co)
        sp.scatter(u[0], u[1], color=co)


if __name__ == '__main__':
    data = generate_dot()
    # np.random.shuffle(data)
    mu, sigma = gmm(data, k=4)
    print(mu)
    plt.show()
    # x = data[:, 0]
    # y = data[:, 1]
    # plt.scatter(x, y)
    # for dot in mu:
    #     plt.scatter(mu[dot][0], mu[dot][1], color="red")
    # plt.axis()
    # plt.title("scatter")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
