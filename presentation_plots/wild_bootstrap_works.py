from stat_test.ar import simulatepm, simulate
from stat_test.quadratic_time import GaussianQuadraticTest

__author__ = 'kcx'
import numpy as np


class GaussianQuadraticForPlotting(GaussianQuadraticTest):

     def compute_stats_for_processes(self,samples,chane_prob):


        U,stat = self.get_statistic_multiple(samples)

        N = samples.shape[0]


        W = simulatepm(N,chane_prob)
        WW = np.outer(W, W)
        st = np.mean(U * WW)
        bootsraped_stat = N * st

        stat = N*np.mean(U)

        return stat,bootsraped_stat



def grad_log_normal( x):
    return -x

me = GaussianQuadraticForPlotting(grad_log_normal)

mc =1000
Xs  = simulate(1500,mc,0.5)

V = []
B = []
for i in range(mc):
    X = Xs[:,i]
    stat,bootsraped_stat = me.compute_stats_for_processes(X,0.1)
    V.append(stat)
    # B.append(bootsraped_stat)

V = np.array(V)
# B = np.array(B)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

new_style = {'grid': False}
plt.rc('axes', **new_style)

# sns.distplot(B, kde=False , label="B")
sns.kdeplot(V, label="V_n", linewidth=4)
axes = plt.gca()
axes.set_xlim([0,23])
y_lims = axes.get_ylim()
q = np.percentile(V,90)
plt.plot([q, q], y_lims,'--b',linewidth=2.5)
plt.legend(prop={'size':16});

sns.plt.show()
