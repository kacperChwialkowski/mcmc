from statsmodels.tsa.stattools import acf

EPSILON = 0.1

__author__ = 'kcx'

from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn

import numpy as np


def grad_log_normal(x):
    return  -x


N = 500



def almost_t_student(N,df,epsilon):
    samples = np.zeros(N)
    xt = 0
    for t in range(N):
        delta = epsilon/2.0*(-(1+df)*xt/(df+xt**2.0)) + np.sqrt(epsilon)*np.random.randn()
        xt = xt + delta
        samples[t] = xt
    return samples



def normal_mild_corr(N):
    samples = np.zeros(N)
    xt = 0
    for t in range(N):
        a = 0.97
        xt = a*xt + np.sqrt(1-a**2)*np.random.randn()
        samples[t] = xt
    return samples


def get_thinning(X,nlags = 50):
    v = acf(X, nlags=nlags, fft=True)
    thinning = np.argmin(np.abs(v - 0.5)) + 1
    return thinning, v[thinning]

TEST_CHAIN_SIZE = 10**6

X = almost_t_student(TEST_CHAIN_SIZE, 5.0, EPSILON)





sgld_thinning, autocorr =  get_thinning(X)
print(autocorr, sgld_thinning)


X = normal_mild_corr(TEST_CHAIN_SIZE)
ar_thinning, autocorr =  get_thinning(X)
print(autocorr,ar_thinning)


P_CHANGE = 0.1

dfs = [1,3,5,7,9]+[1000]
mc_reps = 100
res = np.empty((0,2))

for df in dfs:

    for mc in range(mc_reps):
        print(mc)
        X = almost_t_student(sgld_thinning *N,df, EPSILON)
        X = X[::sgld_thinning]
        me = GaussianQuadraticTest(grad_log_normal)
        U_stat,_ = me.get_statistic_multiple(X)

        pval = me.compute_pvalues_for_processes(U_stat,P_CHANGE)
        res = np.vstack((res,np.array([df, pval])))


for mc in range(mc_reps):
        X = normal_mild_corr(ar_thinning*N)
        X = X[::ar_thinning]
        me = GaussianQuadraticTest(grad_log_normal)
        U_stat,_ = me.get_statistic_multiple(X)
        pval = me.compute_pvalues_for_processes(U_stat,P_CHANGE)
        res = np.vstack((res,np.array([np.Inf, pval])))



df = DataFrame(res)
pr =seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()

exit(0)

X = almost_t_student(1000000,50.0,0.1)
X= X[::15]

import seaborn as sns
sns.set(color_codes=True)
sns.distplot(X);
sns.plt.show()
nlags = 4
v = acf(X, nlags=nlags)
print(v)
print(v[1]**np.arange(nlags))


X = normal_mild_corr(20000)


import seaborn as sns
sns.set(color_codes=True)
sns.distplot(X);
sns.plt.show()
nlags = 5
v = acf(X, nlags=nlags)
print(v)
print(v[1]**np.arange(nlags))

# exit(0)

dfs = range(1, 9, 2)
mc_reps = 100
res = np.empty((0,2))

block = N/N**(0.3)

print(P_CHANGE)

thinning = 15


