from time import time
from sgld_test.gradients_of_likelihood import manual_grad
from sgld_test.mcmc_convergance.cosnt import NUMBER_OF_TESTS, NO_OF_SAMPELS_IN_TEST, CHAIN_SIZE
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

me = GaussianSteinTest(grad_log_pob,5)
size_range = range(5,85,5)
num_pvals = 5
pvals = np.zeros((len(size_range), num_pvals))
size_number =-1
t1 = time()



sample = []
no_chains = NUMBER_OF_TESTS * NO_OF_SAMPELS_IN_TEST
for i in range(no_chains):
    if i % 100 == 0:
        print(i)
        print(time()-t1)
    sample.append(metropolis_hastings(vectorized_log_density, chain_size=CHAIN_SIZE, thinning=1, x_prev=np.random.randn(2)))

sample = np.array(sample)

np.save('samples.npy',sample)
exit(0)

samples = np.array(sample)
pval = []
for ttime in range(85):
    samples_time_ = samples[:, ttime, :]
    assert samples_time_.shape[0] ==100
    pval.append(me.compute_pvalue(samples_time_))




# pvals[size_number,pvs] = me.compute_pvalue(sample)




t2 = time()
print(t2-t1)

import matplotlib.pyplot as plt

plt.plot(pval)
plt.show()

exit(0)

# #
# print(acf(sample[:,1],nlags=20))
#
# with sns.axes_style("white"):
#      sns.jointplot(x=sample[:,1], y=sample[:,0],kind='kde', color="k");
#      sns.plt.show()
#
#
#
#
# me = GaussianSteinTest(grad_log_pob,1)
#
# print(me.compute_pvalue(sample))