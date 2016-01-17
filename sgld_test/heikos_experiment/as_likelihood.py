from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf
from sampplers.austerity import austerity
from sgld_test.bimodal_SGLD import evSGLD
from sgld_test.constants import SIGMA_1, SIGMA_2
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.likelihoods import gen_X, _vector_of_log_likelihoods
from stat_test.quadratic_time import GaussianQuadraticTest

MAGIC_BURNIN_NUMBER = 20

__author__ = 'kcx'
import numpy as np


# np.random.seed(13)
X = gen_X(200)


def vectorized_log_lik(X,theta):
     return _vector_of_log_likelihoods(theta[0],theta[1],X)

def log_density_prior(theta):
    return np.log(norm.pdf(theta[0],0, SIGMA_1)) + np.log(norm.pdf(theta[1],0, SIGMA_2))



THINNING_ESTIMAE = 10**4

sample,evals = austerity(vectorized_log_lik,log_density_prior, X,0.01,batch_size=50,chain_size=THINNING_ESTIMAE, thinning=1, theta_t=np.random.randn(2))



def get_thinning(X,nlags = 50):
    autocorrelation = acf(X, nlags=nlags, fft=True)
    thinning = np.argmin(np.abs(autocorrelation - 0.5)) + 1
    return thinning, autocorrelation


thinning, autocorr =  get_thinning(sample[:,0])

print('the thinning ',thinning)

TEST_SIZE = 500

MAGIC_BURNIN_NUMBER

sample, evals = austerity(vectorized_log_lik,log_density_prior, X,0.01,batch_size=50,chain_size=TEST_SIZE + MAGIC_BURNIN_NUMBER, thinning=thinning, theta_t=np.random.randn(2))

sample = sample[MAGIC_BURNIN_NUMBER:]

assert sample.shape[0] == TEST_SIZE


autocorr = acf(sample[:,0],nlags=3)

print(autocorr)

import seaborn as sns
sns.set(color_codes=True)
with sns.axes_style("white"):
    pr = sns.jointplot(x=sample[:,0], y=sample[:,1], kind="kde", color="k");

    sns.plt.show()


def grad_log_probability(theta):
    return manual_grad(theta[0], theta[1], X)[0]

P_CHANGE =0.1

tester = GaussianQuadraticTest(grad_log_probability)
U_stat, _ = tester.get_statistic_multiple(sample)
p = tester.compute_pvalues_for_processes(U_stat, P_CHANGE)

print('p-value',p)
