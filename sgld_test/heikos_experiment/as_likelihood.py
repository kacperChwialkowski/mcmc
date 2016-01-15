from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf
from sgld_test.bimodal_SGLD import evSGLD
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.likelihoods import gen_X

__author__ = 'kcx'
import numpy as np


TEST_SIZE = 5* 10**5
THINING =1

SGLD_BATCH_SIZE = 1

b = 2.31
a = 0.01584

EPSILON_MIN = 0.002
t = (EPSILON_MIN /a)**(-1.0/0.55)-b
burn_in_epsilons = a*(b+np.arange(t))**(-0.55)



# cov_epsilons = np.append(burn_in_epsilons,np.ones(TEST_SIZE)*EPSILON_MIN)

cov_epsilons = np.ones(TEST_SIZE)*EPSILON_MIN

print(cov_epsilons)

chain_size = len(cov_epsilons)
likelihood_evals = chain_size*SGLD_BATCH_SIZE


# np.random.seed(13)
X = gen_X(100)


sample = evSGLD(manual_grad, grad_log_prior, X, n=SGLD_BATCH_SIZE, epsilons = cov_epsilons,theta = np.random.randn(2) )

# sample = sample[::THINING]

# # estimate size of thinning
# def get_thinning(X,nlags=100):
#     autocorrelation = acf(X, nlags=nlags, fft=True)
#     # find correlation closest to 0.5
#     thinning = np.argmin(np.abs(autocorrelation - 0.5)) + 1
#     return thinning, autocorrelation

print(sample)
import seaborn as sns
sns.set(color_codes=True)
with sns.axes_style("white"):
    pr = sns.jointplot(x=sample[:,0], y=sample[:,1], kind="hex", color="k");
    sns.plt.show()



# thinning, autocorr =  get_thinning(sample[:,0])
# print('thinning for sgld  simulation ', thinning)
# print(autocorr[thinning-10:thinning+10])
#
import matplotlib.pyplot as plt

autocorr = acf(sample[:,0],nlags=1500)

print(autocorr)
# plt.plot(autocorr)


plt.plot(autocorr)
plt.show()

mdl = LinearRegression()
x = np.array(range(100,499))
y = autocorr[x]

x = x[:,np.newaxis]
y = y[:,np.newaxis]

mdl.fit(x,y)

print(mdl.coef_)