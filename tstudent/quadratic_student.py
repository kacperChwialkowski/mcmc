from stat_test.Ustat import GaussianQuadraticTest

__author__ = 'kcx'

import matplotlib
from pandas import DataFrame
import seaborn
from stat_test.random_freq_test import GaussianSteinTest

__author__ = 'kcx'
import numpy as np


def grad_log_normal(x):
    return  -x


N = 300


dfs = range(1, 71, 5)
mc_reps = 20
res = np.empty((0,2))

# for df in dfs:
#
#     for mc in range(mc_reps):
#         print(mc)
#         X = np.random.standard_t(df,N)
#         me = GaussianQuadraticTest(grad_log_normal)
#         pvalue = me.compute_pvalue(X)
#         res = np.vstack((res,np.array([df, pvalue])))

for mc in range(mc_reps):

        X = np.random.randn(N)
        me = GaussianQuadraticTest(grad_log_normal)
        pvalue = me.compute_pvalue(X)
        res = np.vstack((res,np.array([np.Inf, pvalue])))

# import matplotlib.pyplot as plt
# plt.plot(sorted(res[:,1]))
# plt.show()

np.save('results.npy',res)


df = DataFrame(res)
pr =seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()


fig = pr.get_figure()
fig.savefig('../write_up/img/qstudent.pdf')