import pandas as pd
import numpy as np
import KMeans

if __name__ == '__main__':
    data = pd.read_csv("data/iris.csv", header=None)
    data = data.sample(frac=1).reset_index(drop=True)
    attribute = data.iloc[:, :4]
    cluster = data.iloc[:, 4]
    # print(attribute)
    # print(cluster)
    attribute = np.asarray(attribute)
    # print(attribute)
    KMeans.gmm(X=attribute, k=3, drawable=False, cluster=cluster)