
from stat_test.quadratic_time import GaussianQuadraticTest, QuadraticMultiple2


__author__ = 'kcx'
from scipy.spatial.distance import squareform, pdist

import numpy as np
from statsmodels.stats.multitest import multipletests
from stat_test.ar import simulate, simulatepm


# null is Gaussian with id covariance, that is the magic constants here are 
# for the id covariance 
def baringhaus_stat(samples):
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

    AVERAGE_OVER = 100
    BOOTSTRAP_SIZE = 300
    def grad_log_normal( x):
        return -x

    def run_simulation(sample_size):

        for d in [2, 5, 10, 15, 20, 25]:
            samples = []
            for i in range(BOOTSTRAP_SIZE):
                samples.append(baringhaus_stat(np.random.randn(sample_size, d)))
            samples = np.array(samples)
            pvals = []
            pvals2 = []
            for i in range(AVERAGE_OVER):
                X = np.random.randn(SAMPLE_SIZE, d)
                X[:, 0] += np.random.rand(SAMPLE_SIZE)

                T = baringhaus_stat(X)
                pval = len(samples[samples > T]) / BOOTSTRAP_SIZE
                pvals.append(pval)

                me = GaussianQuadraticTest(grad_log_normal)
                qm = QuadraticMultiple2(me)

                p = qm.is_from_null(0.1, X, 0.5)
                pvals2.append(p)

            print('d :', d)
            pvals = np.array(pvals)
            print('baringhaus :', len(pvals[pvals < 0.1]) / AVERAGE_OVER)

            pvals2 = np.array(pvals2)
            print('Stein  :', len(pvals2[pvals2 < 0.1]) / AVERAGE_OVER)

    SAMPLE_SIZE = 500
    run_simulation(SAMPLE_SIZE)
    print("===")
    SAMPLE_SIZE = 1000
    run_simulation(SAMPLE_SIZE)

