from sgld_test.gradients_of_likelihood import manual_grad
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

def grad_log_pob(theta):
    s=[]
    for t in theta:
        s.append( np.sum(manual_grad(t[0],t[1],X),axis=0))
    return np.array(s)

me = GaussianSteinTest(grad_log_pob,1)


size_range = range(1, 60, 4)
num_pvals = 30
pvals = np.zeros((len(size_range), num_pvals))
size_number =-1
for size in size_range:

    size_number +=1
    print(size_number)

    for pvs in range(num_pvals):
        sample = []
        for i in range(400):
            sample.append(metropolis_hastings(vectorized_log_density, chain_size=size, thinning=1, x_prev=np.random.randn(2))[-1])

        sample = np.array(sample)
        pvals[size_number,pvs] = me.compute_pvalue(sample)

np.save('pvals.npy',pvals)


import matplotlib.pyplot as plt

plt.plot(pvals)
plt.show()


# #
# print(acf(sample[:,1],nlags=20))

with sns.axes_style("white"):
     sns.jointplot(x=sample[:,1], y=sample[:,0],kind='kde', color="k");
     sns.plt.show()




me = GaussianSteinTest(grad_log_pob,1)

print(me.compute_pvalue(sample))