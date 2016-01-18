from pandas import DataFrame
import seaborn
from sampplers.MetropolisHastings import metropolis_hastings
from stat_test.quadratic_time import GaussianQuadraticTest, QuadraticMultiple

__author__ = 'kcx'
import numpy as np


def logg(c):
    def log_normal(x):
        return -np.dot(x,x)/c

    return log_normal

def grad_log_dens(x):
    return -x

arr = np.empty((0,2))
for c in [1.0,1.3,2.0,3.0]:
    print('c',c)

    log_normal = logg(c)

    for i in range(23):
        print(i)
        x= metropolis_hastings(log_normal, chain_size=500, thinning=15,x_prev=np.random.randn(2))



        me = GaussianQuadraticTest(grad_log_dens)
        qm = QuadraticMultiple(me)

        accept_null,p_val = qm.is_from_null(0.05, x, 0.1)

        arr = np.vstack((arr, np.array([c,min(p_val)])))



df = DataFrame(arr)
pr = seaborn.boxplot(x=0,y=1,data=df)
seaborn.plt.show()
