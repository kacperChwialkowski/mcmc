from multiprocessing.pool import Pool
from time import time

from sampplers.MetropolisHastings import metropolis_hastings

THINNING = 2

__author__ = 'kcx'
from statsmodels.tsa.stattools import acf
from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn
import numpy as np
from tools.latex_plot_init import plt



SGLD_EPSILON = 0.0478

N = 1000

DEGREES_OF_FREEDOM = [1, np.Inf]
MC_PVALUES_REPS = 32



# The null statistic is that random variables come form normal distibution, so the test statistic takes a gradient of
# logarithm of density of standard normal.
def grad_log_normal(x):
    return -x


def log_normal(x):
    return -x ** 2.0 / 2.0


def grad_log_t_df(df):
    def grad_log_t(x):
        return -(df + 1.0) / 2.0 * np.log(1 + x ** 2 / df)

    return grad_log_t


def gen(N, df, thinning=1):
    log_den = log_normal
    if df < np.Inf:
        log_den = grad_log_t_df(df)
    return metropolis_hastings(log_den, chain_size=N, thinning=thinning, x_prev=np.random.randn(), step=0.5)


# estimate size of thinning
def get_thinning(X, nlags=50):
    autocorrelation = acf(X, nlags=nlags, fft=True)
    thinning = np.argmin(np.abs(autocorrelation - 0.95)) + 1
    return thinning, autocorrelation


# X = gen(TEST_CHAIN_SIZE, np.Inf)
# thinning, autocorr = get_thinning(X)
# print('thinning for AR normal simulation ', thinning, autocorr[thinning])




def get_pval(X, p_change):
    tester = GaussianQuadraticTest(grad_log_normal)
    U_stat, _ = tester.get_statistic_multiple(X)
    return tester.compute_pvalues_for_processes(U_stat, p_change)


def get_pair(df):
    X = gen(N * THINNING, df, thinning=THINNING)
    pval = get_pval(X, p_change=P_CHANGE)
    return [df, pval]


# def map_wrap(sample_size,thinning,tester, p_change):
#     def f(df):
#          return get_pair(sample_size, df, thinning, tester, p_change)
#     return f



P_CHANGE = 0.5




results = map(get_pair, DEGREES_OF_FREEDOM * MC_PVALUES_REPS)


np.save('results_bad.npy', list(results))

P_CHANGE = 0.02



results = map(get_pair, DEGREES_OF_FREEDOM * MC_PVALUES_REPS)


np.save('results_good.npy', list(results))
