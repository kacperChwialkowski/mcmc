from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import acf
from sampplers.austerity import austerity
from sgld_test.bimodal_SGLD import evSGLD
from sgld_test.constants import SIGMA_1, SIGMA_2
from sgld_test.gradients_of_likelihood import manual_grad, grad_log_prior
from sgld_test.likelihoods import gen_X, _vector_of_log_likelihoods

__author__ = 'kcx'
import numpy as np


TEST_SIZE =  10**4
THINING =1

# np.random.seed(13)
X = gen_X(2000)


def vectorized_log_lik(X,theta):
     return _vector_of_log_likelihoods(theta[0],theta[1],X)

def log_density_prior(theta):
    return np.log(norm.pdf(theta[0],0, SIGMA_1)) + np.log(norm.pdf(theta[1],0, SIGMA_2))



sample = austerity(vectorized_log_lik,log_density_prior, X,0.01,batch_size=50,chain_size=20*1000, thinning=1, theta_t=np.random.randn(2))



print(sample)
import seaborn as sns
sns.set(color_codes=True)
with sns.axes_style("white"):
    pr = sns.jointplot(x=sample[:,0], y=sample[:,1], kind="hex", color="k");
    sns.plt.show()



def get_thinning(X,nlags = 50):
    autocorrelation = acf(X, nlags=nlags, fft=True)
    # find correlation closest to given v
    thinning = np.argmin(np.abs(autocorrelation - 0.85)) + 1
    return thinning, autocorrelation


thinning, autocorr =  get_thinning(sample[:,0])
print('thinning for sgld  simulation ', thinning)
print(autocorr[thinning-10:thinning+10])
#
import matplotlib.pyplot as plt

autocorr = acf(sample[:,0],nlags=1500)

print(autocorr)


plt.plot(autocorr)
plt.show()

mdl = LinearRegression()
x = np.array(range(100,499))
y = autocorr[x]

x = x[:,np.newaxis]
y = y[:,np.newaxis]

mdl.fit(x,y)

print(mdl.coef_)