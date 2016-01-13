from statsmodels.tsa.stattools import acf
from sgld_test.bimodal_SGLD import evSGLD
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.likelihoods import gen_X

__author__ = 'kcx'
import numpy as np


TEST_SIZE = 10000*10


SGLD_BATCH_SIZE = 20.0

b = 2.31
a = 0.01584

EPSILON_MIN = 0.001
t = (EPSILON_MIN /a)**(-1.0/0.55)-b
burn_in_epsilons = a*(b+np.arange(t))**(-0.55)
epsilons = np.append(burn_in_epsilons,np.ones(TEST_SIZE)*EPSILON_MIN)

chain_size = len(epsilons)
likelihood_evals = chain_size*SGLD_BATCH_SIZE



np.random.seed(13)
X = gen_X(500)


sample = evSGLD(manual_grad, grad_log_prior, X, n=1, epsilons = epsilons,theta = np.random.randn(2) )

sample = sample[-TEST_SIZE:]

print(sample.shape)

sample = sample[::10]

print(sample.shape)

print(acf(sample[:,0],nlags=20))
