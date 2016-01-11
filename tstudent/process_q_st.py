__author__ = 'kcx'

from stat_test.quadratic_time import GaussianQuadraticTest
from pandas import DataFrame
import seaborn

import numpy as np


def grad_log_normal(x):
    return  -x


N = 500


dfs = range(1, 11, 2)
mc_reps = 100
res = np.empty((0,2))

# for df in dfs:
#
#     for mc in range(mc_reps):
#         print(mc)
#         X = np.random.standard_t(df,N)
#         me = GaussianQuadraticTest(grad_log_normal)
#         U_stat,_ = me.get_statistic_multiple(X)
#         pval = me.compute_pvalues_for_processes(U_stat,0.5)
#         res = np.vstack((res,np.array([df, pval])))
#
# for mc in range(mc_reps):
#
#         X = np.random.randn(N)
#         me = GaussianQuadraticTest(grad_log_normal)
#         U_stat,_ = me.get_statistic_multiple(X)
#         pval = me.compute_pvalues_for_processes(U_stat,0.5)
#         res = np.vstack((res,np.array([np.Inf, pval])))
#
#
# np.save('results.npy',res)
#
#
# df = DataFrame(res)
# pr =seaborn.boxplot(x=0,y=1,data=df)
# seaborn.plt.show()
#
#
# fig = pr.get_figure()
# fig.savefig('../write_up/img/pqstudent.pdf')



def correlatet_t(X,N):

    fc = np.sign(np.random.randn(N))
    for i in range(1,N):
        if fc[i]>0:
            X[i] = X[i-1]
    return X

dfs = range(1, 4, 2)
mc_reps = 100
res = np.empty((0,2))



for df in dfs:

    for mc in range(mc_reps):
        print(mc)
        X = np.random.standard_t(df,N)
        X = correlatet_t(X,N)
        me = GaussianQuadraticTest(grad_log_normal)
        U_stat,_ = me.get_statistic_multiple(X)
        pval = me.compute_pvalues_for_processes(U_stat,0.99)
        res = np.vstack((res,np.array([df, pval])))

for mc in range(mc_reps):

        X = np.random.randn(N)
        X = correlatet_t(X,N)
        me = GaussianQuadraticTest(grad_log_normal)
        U_stat,_ = me.get_statistic_multiple(X)
        pval = me.compute_pvalues_for_processes(U_stat,0.99)
        res = np.vstack((res,np.array([np.Inf, pval])))



df = DataFrame(res)
pr =seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()
