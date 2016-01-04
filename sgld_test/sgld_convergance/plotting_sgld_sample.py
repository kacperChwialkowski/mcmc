from time import time
from sgld_test.bimodal_SGLD import one_sample_SGLD

from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.likelihoods import gen_X
from stat_test.stationary_distribution import GaussianSteinTest
import numpy as np



np.random.seed(32)
X = gen_X(400)

def grad_log_pob(theta):
    s=[]
    for t in theta:
        s.append( np.sum(manual_grad(t[0],t[1],X),axis=0))
    return np.array(s)

me = GaussianSteinTest(grad_log_pob,1)



me = GaussianSteinTest(grad_log_pob,5)
size_range = range(5,166,10)
num_pvals = 50
pvals = np.zeros((len(size_range), num_pvals))
size_number =-1
t1 = time()
for size in size_range:

    size_number +=1
    print(size_number)

    for pvs in range(num_pvals):
        sample = []
        for i in range(400):
            sample.append(one_sample_SGLD(manual_grad, grad_log_prior, X, n=5, chain_size=size))

        sample = np.array(sample)
        pvals[size_number,pvs] = me.compute_pvalue(sample)

np.save('pvals.npy',pvals)
t2 = time()
print(t2-t1)


# def wrap(i):
#     return one_sample_SGLD(manual_grad, grad_log_prior, X, n=5, chain_size=500)
#
# pool = multiprocessing.Pool(8)
# sample = pool.map(wrap, range(NNN))
#
#
#
# sample = np.array(sample)

#
#
#
# with sns.axes_style("white"):
#     sns.jointplot(x=sample[:,1], y=sample[:,0],kind='scatter', color="k");
#     sns.plt.show()
