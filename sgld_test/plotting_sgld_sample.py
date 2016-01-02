from autograd import grad
from numpy.testing import assert_almost_equal
from statsmodels.tsa.stattools import acf
from sgld_test.bimodal_SGLD import SGLD
from sgld_test.test import gen_X, _log_lik, log_probability
import seaborn as sns; sns.set(color_codes=True)
from scipy.stats import norm
import autograd.numpy as np   # Thinly-wrapped version of Numpy

import matplotlib.pyplot as plt


SIGMA_x = np.sqrt(2.0)
SIGMA_1 = np.sqrt(10)
SIGMA_2 = 1.


def scalar_log_lik(theta_1,theta_2,x):
    arg = (x - theta_1)
    lik1 = 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( - np.dot(arg,arg)/(2*SIGMA_x**2) )
    arg = (x - theta_1 - theta_2)
    lik2 = 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( - np.dot(arg,arg)/(2*SIGMA_x**2) )

    return np.log(0.5*lik1+0.5*lik2)

assert_almost_equal(scalar_log_lik(1.0,2.0,3.0),_log_lik(1.0,2.0,3.0))


def grad_log_prior(theta):
    return theta/[SIGMA_1,SIGMA_2]


grad_the_log_density_x = grad(scalar_log_lik,0)
grad_the_log_density_y = grad(scalar_log_lik,1)

# I was always good at pen and paper gradient calculations ! hahahahaha! I want to kill myself
assert_almost_equal(grad_the_log_density_x(1.,2.,3.),0.268941,decimal=4)
assert_almost_equal(grad_the_log_density_y(1.,3.,3.),-0.339589,decimal=4)

def grad_the_log_density(theta,x):
    x_derivative = grad_the_log_density_x(theta[0], theta[1], x)
    y_derivative = grad_the_log_density_y(theta[0], theta[1], x)
    return np.array( [x_derivative, y_derivative])


X = gen_X(100)
def vectorized_log_density(theta):
     return log_probability(theta,X)



for i in range()
    sample,A = SGLD(grad_the_log_density,grad_log_prior, X,n=5,log_density=vectorized_log_density,thinning=1,chain_size=500)

print(acf(sample[:,1],nlags=10))

plt.plot(np.convolve(A, np.ones((500,))/500, mode='valid'))
plt.show()


sample = sample[100:]



print(sample)

with sns.axes_style("white"):
     sns.jointplot(x=sample[:,1], y=sample[:,0],kind='scatter', color="k");
     sns.plt.show()
