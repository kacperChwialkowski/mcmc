import multiprocessing
import seaborn as sns

from sgld_test.bimodal_SGLD import one_sample_SGLD

from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.test import gen_X
from stat_test.stationary_distribution import GaussianSteinTest


sns.set(color_codes=True)
import autograd.numpy as np


X = gen_X(100)

NNN = 100


def wrap(i):
    return one_sample_SGLD(manual_grad, grad_log_prior, X, n=5, chain_size=500, theta=4 * np.random.rand(2) - 2)

pool = multiprocessing.Pool(8)
sample = pool.map(wrap, range(NNN))



sample = np.array(sample)




with sns.axes_style("white"):
    sns.jointplot(x=sample[:,1], y=sample[:,0],kind='scatter', color="k");
    sns.plt.show()

def grad_for_single_theta(t1,t2):
    return manual_grad(t1,t2,X)

def grad_log_pob(theta):
    s=[]
    for t in theta:
        s.append( np.sum(manual_grad(t[0],t[1],X),axis=0))
    return np.array(s)

me = GaussianSteinTest(grad_log_pob,1)

print(me.compute_pvalue(sample))