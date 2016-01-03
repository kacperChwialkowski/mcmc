from statsmodels.tsa.stattools import acf
from sgld_test.plotting_sgld_sample import manual_grad
from sgld_test.test import gen_X, log_probability
import seaborn as sns;
from sampplers.MetropolisHastings import metropolis_hastings
import numpy as np
from stat_test.stationary_distribution import GaussianSteinTest

sns.set(color_codes=True)
__author__ = 'kcx'

np.random.seed(32)
X = gen_X(400)

def vectorized_log_density(theta):
     return log_probability(theta,X)

sample = []
for i in range(400):
    print(i)
    sample.append(metropolis_hastings(vectorized_log_density, chain_size=400, thinning=1, x_prev=np.random.randn(2))[-1])

sample = np.array(sample)

# #
print(acf(sample[:,1],nlags=20))

with sns.axes_style("white"):
     sns.jointplot(x=sample[:,1], y=sample[:,0],kind='kde', color="k");
     sns.plt.show()



def grad_log_pob(theta):
    s=[]
    for t in theta:
        s.append( np.sum(manual_grad(t[0],t[1],X),axis=0))
    return np.array(s)

me = GaussianSteinTest(grad_log_pob,1)

print(me.compute_pvalue(sample))