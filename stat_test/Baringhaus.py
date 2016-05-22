
from stat_test.quadratic_time import GaussianQuadraticTest, QuadraticMultiple2

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
    # samples = []
    # for i in range(300):
    #     samples.append(stat(np.random.randn(200,2)))
    # samples1 = []
    # for i in range(300):
    #      samples1.append(stat(np.random.randn(1000,2)))

    def grad_log_normal( x):
        return -x

    SAMPLE_SIZE = 500
    # n = 400
    sim = []
    for d in [70]:
        samples = []
        for i in range(300):
            samples.append(stat(np.random.randn(SAMPLE_SIZE,d)))
        samples = np.array(samples)
        pvals =[]
        pvals2 =[]
        for i in range(100):

            X = np.random.randn(SAMPLE_SIZE,d)
            X[:,0] += np.random.rand(SAMPLE_SIZE)

            T = stat(X)
            pval = len(samples[samples>T])/300
            pvals.append(pval)

            # me = GaussianQuadraticTest(grad_log_normal)
            # qm = QuadraticMultiple2(me)
            #
            # p = qm.is_from_null(0.1, X, 0.5)
            # pvals2.append(p)

        print('d :',d )
        pvals = np.array(pvals)
        print('them :',len(pvals[pvals < 0.1])/100)

        # pvals2 = np.array(pvals2)
        # print('us :',len(pvals2[pvals2 < 0.1])/100)

        #
        # import matplotlib.pyplot as plt
        # plt.plot(sorted(pvals))
        # plt.show()


    SAMPLE_SIZE = 1000
    # n = 400
    sim = []
    for d in [2,5,10,15,20,25,30]:
        samples = []
        for i in range(300):
            samples.append(stat(np.random.randn(SAMPLE_SIZE,d)))
        samples = np.array(samples)
        pvals =[]
        pvals2 =[]
        for i in range(100):

            X = np.random.randn(SAMPLE_SIZE,d)
            X[:,0] += np.random.rand(SAMPLE_SIZE)

            T = stat(X)
            pval = len(samples[samples>T])/300
            pvals.append(pval)

            me = GaussianQuadraticTest(grad_log_normal)
            qm = QuadraticMultiple2(me)

            p = qm.is_from_null(0.1, X, 0.5)
            pvals2.append(p)

        print('d :',d )
        pvals = np.array(pvals)
        print('them :',len(pvals[pvals < 0.1])/100)

        pvals2 = np.array(pvals2)
        print('us :',len(pvals2[pvals2 < 0.1])/100)

        #
        # import matplotlib.pyplot as plt
        # plt.plot(sorted(pvals))
        # plt.show()

