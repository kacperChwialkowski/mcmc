from numpy.ma.testutils import assert_almost_equal
from scipy.stats import norm
import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
import seaborn as sns;
from sampplers.MetropolisHastings import metropolis_hastings
# from sampplers.SGLD import sgld

sns.set(color_codes=True)
SIGMA_x = np.sqrt(2.0)
SIGMA_1 = np.sqrt(10)
SIGMA_2 = 1.


def log_probability(theta,x):

    theta_1 = theta[0]
    theta_2 = theta[1]
    return _log_probability(theta_1,theta_2,x)

def _log_probability(theta_1,theta_2,x):
    lik = 0.5*norm.pdf(x, theta_1, SIGMA_x)+ 0.5*norm.pdf(x, theta_1+theta_2, SIGMA_x)

    log_lik = np.sum(np.log(lik))

    log_prior = np.log(norm.pdf(theta_1,0, SIGMA_1) + norm.pdf(theta_2,0, SIGMA_2))

    return (log_lik+log_prior)

def same(theta_1,theta_2,x):
    return np.log(0.5*norm.pdf(x, theta_1, SIGMA_x)+ 0.5*norm.pdf(x, theta_1+theta_2, SIGMA_x))


def log_probability(theta_1,theta_2,x):
    arg = (x - theta_1)
    lik1 = 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( - np.dot(arg,arg)/(2*SIGMA_x**2) )
    arg = (x - theta_1 - theta_2)
    lik2 = 1.0/np.sqrt(2*SIGMA_x**2*np.pi)*np.exp( - np.dot(arg,arg)/(2*SIGMA_x**2) )

    return np.log(0.5*lik1+0.5*lik2)

assert_almost_equal( same(1,2,3), log_probability(1,2,3))



def grad_log_prior(theta):
    return theta/[SIGMA_1,SIGMA_2]

def gen_X(n):
    res = []
    true_theta_1 = -1
    true_theta_2 = 1.
    for _ in range(n):
        coin  = np.random.rand()
        if coin < 0.5:
            add = np.random.randn()*SIGMA_x+true_theta_1
        else:
            add = np.random.randn()*SIGMA_x+true_theta_1+true_theta_2

        res.append(add)

    return np.array(res)


# print(np.concatenate(,Y[:,:,np.newaxis]))
X = gen_X(300)


def vectorized_log_density(theta):
    return log_probability(theta,X)

grad_the_log_density_x = grad(log_probability,0)
grad_the_log_density_y = grad(log_probability,1)

grad_the_log_density_x(1.,2.,3.)

#
# sample = metropolis_hastings(the_log_density, chain_size=1000, thinning=1, x_prev=np.random.randn(2),a=1.0,b=1.0,gamma=0.55)
#
# print(sample)
#
# with sns.axes_style("white"):
#      sns.jointplot(x=sample[:,0], y=sample[:,1], kind="kde", color="k",xlim=[-3,2],ylim=[-2,3]);
#      sns.plt.show()
#


