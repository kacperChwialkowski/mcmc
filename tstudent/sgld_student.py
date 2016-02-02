from sampplers.MetropolisHastings import metropolis_hastings

__author__ = 'kcx'
from statsmodels.tsa.stattools import acf
from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn
import numpy as np
from tools.latex_plot_init import plt

SGLD_EPSILON = 0.0478


N = 1000

DEGREES_OF_FREEDOM = [1,np.Inf]
MC_PVALUES_REPS = 10
TEST_CHAIN_SIZE = 2 * 10 ** 6


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




def gen(N,df):
    log_den =  log_normal
    if df <np.Inf:
        log_den = grad_log_t_df(df)
    return  metropolis_hastings(log_den, chain_size=N, thinning=1, x_prev=np.random.randn(), step=0.5)

# estimate size of thinning
def get_thinning(X, nlags=50):
    autocorrelation = acf(X, nlags=nlags, fft=True)
    thinning = np.argmin(np.abs(autocorrelation - 0.95)) + 1
    return thinning, autocorrelation


X = gen(TEST_CHAIN_SIZE, np.Inf)
ar_thinning, autocorr = get_thinning(X)
print('thinning for AR normal simulation ', ar_thinning, autocorr[ar_thinning])

X = gen(TEST_CHAIN_SIZE,100.0)
sgld_thinning, autocorr = get_thinning(X)
print('thinning for sgld t-student simulation ', sgld_thinning, autocorr[sgld_thinning])



tester = GaussianQuadraticTest(grad_log_normal)

# This stupid function takes global argument
def get_pval(X, tester):
    U_stat, _ = tester.get_statistic_multiple(X)
    return tester.compute_pvalues_for_processes(U_stat, P_CHANGE)


P_CHANGE = 0.5
results = []
for df in DEGREES_OF_FREEDOM:
    print(df)
    for mc in range(MC_PVALUES_REPS):
        if mc % 32 == 0:
            print(' ', 100.0 * mc / MC_PVALUES_REPS, '%')
        X = gen(sgld_thinning * N, df)
        X = X[::sgld_thinning]
        pval = get_pval(X, tester)
        results.append([df, pval])


np.save('results_bad.npy', results)


P_CHANGE = 0.02
results = np.empty((0, 2))
for df in DEGREES_OF_FREEDOM:
    print(df)
    for mc in range(MC_PVALUES_REPS):
        if mc % 32 == 0:
            print(' ', 100.0 * mc / MC_PVALUES_REPS, '%')
        X = gen(sgld_thinning * N, df)
        X = X[::sgld_thinning]
        pval = get_pval(X, tester)
        results = np.vstack((results, np.array([df, pval])))


np.save('results_good.npy', results)
