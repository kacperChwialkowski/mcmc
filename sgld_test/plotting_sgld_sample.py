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

def lik_2(theta_1,theta_2,x):
    arg = (x - theta_1 - theta_2)
    return 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( - arg**2/(2*SIGMA_x**2) )

def lik_1(theta_1,theta_2,x):
    arg = (x - theta_1)
    return 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( -  arg**2/(2*SIGMA_x**2) )

def lik_mix(theta_1,theta_2,x):
    return (lik_1(theta_1,theta_2,x)+lik_2(theta_1,theta_2,x))/2.0


def manual_grad_y(theta1,theta2,x):
    nr = lik_2(theta1,theta2,x)/lik_mix(theta1,theta2,x)
    b= nr*(-theta1 -theta2+x)/(2*SIGMA_x**2)
    return b


def manual_grad_y(theta1,theta2,x):
    nr = lik_2(theta1,theta2,x)/lik_mix(theta1,theta2,x)
    b= nr*(-theta1 -theta2+x)/(2*SIGMA_x**2)
    return b

def manual_grad_x(theta1,theta2,x):
    a = lik_1(theta1,theta2,x)*(-theta1 +x)/(2*SIGMA_x**2)

    b = lik_2(theta1,theta2,x)*(-theta1 -theta2+x)/(2*SIGMA_x**2)
    return (a+b)/lik_mix(theta1,theta2,x)

assert_almost_equal(grad_the_log_density_y(1.,3.,3.),manual_grad_y(1.,3.,3.),decimal=4)
assert_almost_equal(grad_the_log_density_x(1.,3.,3.),manual_grad_x(1.,3.,3.),decimal=4)


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







def grad_the_log_density(theta,x):
    x_derivative = grad_the_log_density_x(theta[0], theta[1], x)
    y_derivative = grad_the_log_density_y(theta[0], theta[1], x)
    return np.array( [x_derivative, y_derivative])

np.random.seed(0)
X = np.array([13.])
def vectorized_log_density(theta):
     return log_probability(theta,X)

r1= slow_one_sample_SGLD(grad_the_log_density,grad_log_prior,X,n=1,chain_size=300,theta=np.array([1.,1.3]))
np.random.seed(0)
X = np.array([13.])
def vectorized_log_density(theta):
     return log_probability(theta,X)
r2 = one_sample_SGLD(manual_grad,grad_log_prior,X,n=1,chain_size=300,theta=np.array([1.,1.3]))

assert_almost_equal(r1,r2)



X = gen_X(100)

#
# sample = []
# for i in range(500):
#     print(i)
#     sample.append(one_sample_SGLD(manual_grad,grad_log_prior,X,n=5,chain_size=500))

NNN = 1000

def wrap(i):
    return one_sample_SGLD(manual_grad,grad_log_prior,X,n=5,chain_size=500,theta=4*np.random.rand(2)-2)

# pool = multiprocessing.Pool(8)
# sample = pool.map(wrap, range(NNN))
#
#
#
# sample = np.array(sample)
#
#
#
#
# with sns.axes_style("white"):
#      sns.jointplot(x=sample[:,1], y=sample[:,0],kind='scatter', color="k");
#      sns.plt.show()
#
# def grad_for_single_theta(t1,t2):
#     return manual_grad(t1,t2,X)
#
# def grad_log_pob(theta):
#     s=[]
#     for t in theta:
#         s.append( np.sum(manual_grad(t[0],t[1],X),axis=0))
#     return np.array(s)
#
# me = GaussianSteinTest(grad_log_pob,1)
#
# print(me.compute_pvalue(sample))