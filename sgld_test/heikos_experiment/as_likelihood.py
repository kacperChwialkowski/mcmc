from scipy.stats import norm
from statsmodels.tsa.stattools import acf

from sampplers.MetropolisHastings import metropolis_hastings
from sampplers.austerity import austerity
from sgld_test.bimodal_SGLD import evSGLD
from sgld_test.constants import SIGMA_1, SIGMA_2
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.likelihoods import gen_X, _vector_of_log_likelihoods, log_probability
from stat_test.linear_time import GaussianSteinTest
from stat_test.quadratic_time import GaussianQuadraticTest, QuadraticMultiple

MAGIC_BURNIN_NUMBER = 200

__author__ = 'kcx'
import numpy as np


# np.random.seed(13)
SAMPLE_SIZE = 400
X = gen_X(SAMPLE_SIZE)


def vectorized_log_lik(X,theta):
     return _vector_of_log_likelihoods(theta[0],theta[1],X)

def log_density_prior(theta):
    return np.log(norm.pdf(theta[0],0, SIGMA_1)) + np.log(norm.pdf(theta[1],0, SIGMA_2))


# for epsilon in [0.0001,0.001,0.01,0.1,0.2]:
# THINNING_ESTIMAE = 10**4
#
# sample,evals = austerity(vectorized_log_lik,log_density_prior, X,0.01,batch_size=SAMPLE_SIZE, chain_size=THINNING_ESTIMAE, thinning=1, theta_t=np.random.randn(2))
#
#
#
# def get_thinning(X,nlags = 50):
#     autocorrelation = acf(X, nlags=nlags, fft=True)
#     thinning = np.argmin(np.abs(autocorrelation - 0.5)) + 1
#     return thinning, autocorrelation
#
#
# thinning, autocorr =  get_thinning(sample[:,0])
#
#
# print('the thinning ',thinning)
# print('autocorr', autocorr)

thinning=51

TEST_SIZE = 500


sample, evals = austerity(vectorized_log_lik,log_density_prior, X,0.001,batch_size=50,chain_size=TEST_SIZE + MAGIC_BURNIN_NUMBER, thinning=thinning, theta_t=np.random.randn(2))

# def vectorized_log_density(theta):
#      return log_probability(theta,X)
#
# sample = metropolis_hastings(vectorized_log_density, chain_size=TEST_SIZE+MAGIC_BURNIN_NUMBER, thinning=thinning, x_prev=np.random.randn(2))

sample = sample[MAGIC_BURNIN_NUMBER:]

assert sample.shape[0] == TEST_SIZE


autocorr = acf(sample[:,0],nlags=3)

print(autocorr)
#
# import seaborn as sns
# sns.set(color_codes=True)
#
# pr = sns.jointplot(x=sample[:,0], y=sample[:,1], kind="kde", color="k")
#
# sns.plt.show()


def grad_log_pob(t):
    a = np.sum(manual_grad(t[0],t[1],X),axis=0) + grad_log_prior(t)
    return a

def grad_log_pob_1(t):
    return grad_log_pob(t)[1]

P_CHANGE =0.1

me = GaussianQuadraticTest(grad_log_pob)
qm = QuadraticMultiple(me)

me.get_statisitc()


reject, p = qm.is_from_null(0.05, sample, 0.1)
print('====     p-value',p)
print('====     reject',reject)

def grad_log_pob_v(theta):
    s=[]
    for t in theta:
        s.append( np.sum(manual_grad(t[0],t[1],X),axis=0))
    return np.array(s)

me = GaussianSteinTest(grad_log_pob_v,1)


pval = me.compute_pvalue(sample)
print('====     p-value',pval)




