
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

    def grad_log_normal( x):
        return -x

    def run_simulation(sample_size, bootstrap_size=300, average_over=100):

        for d in [2, 5, 10, 15, 20, 25]:
            samples = []
            for i in range(bootstrap_size):
                samples.append(baringhaus_stat(np.random.randn(sample_size, d)))
            samples = np.array(samples)
            pvals_brainghaus = []
            pvals_stein = []
            for i in range(average_over):
                X = np.random.randn(sample_size, d)
                X[:, 0] += np.random.rand(sample_size)

                T = baringhaus_stat(X)
                pval = len(samples[samples > T]) / bootstrap_size
                pvals_brainghaus.append(pval)

                me = GaussianQuadraticTest(grad_log_normal)
                qm = QuadraticMultiple2(me)

                p = qm.is_from_null(0.1, X, 0.5)
                pvals_stein.append(p)

            print('d :', d)
            pvals_brainghaus = np.array(pvals_brainghaus)
            print('baringhaus :', len(pvals_brainghaus[pvals_brainghaus < 0.1]) / average_over)

            pvals_stein = np.array(pvals_stein)
            print('Stein  :', len(pvals_stein[pvals_stein < 0.1]) / average_over)

    sample_size = 500
    run_simulation(sample_size)
    print("===")
    sample_size = 1000
    run_simulation(sample_size)

