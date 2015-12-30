import matplotlib
from pandas import DataFrame
import seaborn
from stat_test.stationary_distribution import GaussianSteinTest

__author__ = 'kcx'
import numpy as np


def grad_log_normal(x):
    return  -x

m=5




dfs = range(1, 71, 5)
mc_reps = 150
res = np.empty((0,2))

for df in dfs:
    for mc in range(mc_reps):

        X = np.random.standard_t(df,1500)
        me = GaussianSteinTest(grad_log_normal,m)
        pvalue = me.compute_pvalue(X)
        res = np.vstack((res,np.array([df, pvalue])))

for mc in range(mc_reps):

        X = np.random.randn(10000)
        me = GaussianSteinTest(grad_log_normal,m)
        pvalue = me.compute_pvalue(X)
        res = np.vstack((res,np.array([np.Inf, pvalue])))

# import matplotlib.pyplot as plt
# plt.plot(sorted(res[:,1]))
# plt.show()

np.save('results.npy',res)


df = DataFrame(res)

seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()