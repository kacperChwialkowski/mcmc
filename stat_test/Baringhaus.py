
__author__ = 'kcx'
from scipy.spatial.distance import squareform, pdist

import numpy as np
from statsmodels.stats.multitest import multipletests
from stat_test.ar import simulate, simulatepm


# null are only Gaussians with id covariance
# class BaringhausTest:


def stat(samples):
    Y = samples - np.mean(samples)
    n = Y.shape[0]
    d = Y.shape[1]

    R =  squareform(pdist(Y, 'euclidean'))**2
    R2 = np.linalg.norm(Y,axis=1)**2

    T1 = np.sum( np.exp(-0.5 *R))/n
    T2 = -2.0**(1.0-d/2.0)*np.sum(np.exp(-0.25*R2))
    T3 = n*3.0**(-d/2.0)
    return T1+T2+T3



if __name__ == "__main__":
    samples = []
    for i in range(300):
        samples.append(stat(np.random.randn(200,2)))
    samples1 = []
    for i in range(300):
         samples1.append(stat(np.random.randn(1000,2)))

    print(samples)
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)

    sns.kdeplot(np.array(samples))
    sns.kdeplot(np.array(samples1))

    sns.plt.show()
