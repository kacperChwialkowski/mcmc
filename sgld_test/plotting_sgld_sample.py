from autograd import grad
from numpy.testing import assert_almost_equal
from sgld_test.test import gen_X, log_probability
import seaborn as sns; sns.set(color_codes=True)
from scipy.stats import norm
import autograd.numpy as np   # Thinly-wrapped version of Numpy

SIGMA_x = np.sqrt(2.0)
SIGMA_1 = np.sqrt(10)
SIGMA_2 = 1.




def log_conditional_scipy(theta_1,theta_2,x):
    return np.log(0.5*norm.pdf(x, theta_1, SIGMA_x)+ 0.5*norm.pdf(x, theta_1+theta_2, SIGMA_x))


def scalar_log_conditional(theta_1,theta_2,x):
    arg = (x - theta_1)
    lik1 = 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( - np.dot(arg,arg)/(2*SIGMA_x**2) )
    arg = (x - theta_1 - theta_2)
    lik2 = 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( - np.dot(arg,arg)/(2*SIGMA_x**2) )

    return np.log(0.5*lik1+0.5*lik2)


assert_almost_equal( log_conditional_scipy(1,2,3), scalar_log_conditional(1,2,3))



def grad_log_prior(theta):
    return theta/[SIGMA_1,SIGMA_2]


grad_the_log_density_x = grad(scalar_log_conditional,0)
grad_the_log_density_y = grad(scalar_log_conditional,1)

# I was always good at pen and paper gradient calculations ! hahahahaha! I want to kill myself
assert_almost_equal(grad_the_log_density_x(1.,2.,3.),0.268941,decimal=4)
assert_almost_equal(grad_the_log_density_y(1.,3.,3.),-0.339589,decimal=4)

def grad_the_log_density(theta,x):
    kurwa = grad_the_log_density_x(theta[0], theta[1], x)
    mac = grad_the_log_density_y(theta[0], theta[1], x)
    return np.array( [kurwa, mac])

X = gen_X(200)

def vectorized_log_density(theta):
     return log_probability(theta,X)


#
#

sample = SGLD(vectorized_log_density, grad_the_log_density,grad_log_prior, X,1,chain_size=100, thinning=2)


print(sample)

with sns.axes_style("white"):
     sns.jointplot(x=sample[:,1], y=sample[:,0],kind='kde', color="k");
     sns.plt.show()
