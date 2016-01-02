from autograd import grad
from numpy.testing import assert_almost_equal
from sgld_test.bimodal_SGLD import SGLD
from sgld_test.test import gen_X, log_probability, _log_lik
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



theta1 = np.arange(-2, 2 ,0.1)
the_len = len(theta1)
theta2 = np.arange(-2, 2, 0.1)
theta1, theta2 = np.meshgrid(theta1, theta2)
Dx = np.copy(theta1)
Dy = np.copy(theta1)

Xsample = gen_X(400)


for i in range(the_len):
    for j in range(the_len):
        print(i,j)
        th = np.array([theta1[i, j], theta2[i, j]])
        stupid_sum=np.array([0.0,0.0])
        sub = np.random.choice(Xsample, 4)

        for data_point in sub:
            stupid_sum = stupid_sum+ grad_the_log_density(th,data_point)
        # grad_log_prior(th)
        grad =  + stupid_sum
        Dx[i,j] = grad[0]
        Dy[i,j] = grad[1]


plt.figure()
CS = plt.streamplot(theta1, theta2, Dx, Dy, density=[0.5, 1])

plt.show()


