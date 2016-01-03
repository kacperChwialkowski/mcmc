__author__ = 'kcx'

from autograd import grad
import multiprocessing
from numpy.testing import assert_almost_equal
from statsmodels.tsa.stattools import acf
from sgld_test.bimodal_SGLD import SGLD, one_sample_SGLD, slow_one_sample_SGLD
from sgld_test.test import gen_X, _log_lik, log_probability
import seaborn as sns;
from stat_test.stationary_distribution import GaussianSteinTest

sns.set(color_codes=True)
from scipy.stats import norm
import autograd.numpy as np   # Thinly-wrapped version of Numpy


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

grad_the_log_density_x = grad(scalar_log_lik,0)
grad_the_log_density_y = grad(scalar_log_lik,1)

# Compare with versions from mathematica
assert_almost_equal(grad_the_log_density_x(1.,2.,3.),0.268941,decimal=4)
assert_almost_equal(grad_the_log_density_y(1.,3.,3.),-0.339589,decimal=4)

# Manual log likelihood

def lik_2(theta_1,theta_2,x):
    arg = (x - theta_1 - theta_2)
    return 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( - arg**2/(2*SIGMA_x**2) )

def lik_1(theta_1,theta_2,x):
    arg = (x - theta_1)
    return 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( -  arg**2/(2*SIGMA_x**2) )

def lik_mix(theta_1,theta_2,x):
    return (lik_1(theta_1,theta_2,x)+lik_2(theta_1,theta_2,x))/2.0




def manual_grad(theta1,theta2,x):
    lik1 = lik_1(theta1,theta2,x)
    lik2 = lik_2(theta1,theta2,x)
    mix = lik_mix(theta1,theta2,x)
    twoSigmaSquare = (2 * SIGMA_x ** 2)
    pgx = lik1*(-theta1 +x)/ twoSigmaSquare
    pgy = lik2*(-theta1 -theta2+x)/ twoSigmaSquare
    gx = (pgx+pgy)/mix
    gy = pgy/mix
    return np.array([gx,gy]).T

assert_almost_equal(grad_the_log_density_x(1.,3.,3.),manual_grad(1.,3.,3.)[0])
assert_almost_equal(grad_the_log_density_y(1.,3.,3.),manual_grad(1.,3.,3.)[1])

assert_almost_equal(manual_grad(1.,3.,np.array([1.0,3.0,5.0]))[-1],manual_grad(1.,3.,5.))
