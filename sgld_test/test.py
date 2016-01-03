from scipy.stats import norm
import autograd.numpy as np   # Thinly-wrapped version of Numpy
# from sampplers.SGLD import sgld

SIGMA_x = np.sqrt(2.0)
SIGMA_1 = np.sqrt(10)
SIGMA_2 = 1.


def log_probability(theta,x):
    theta_1 = theta[0]
    theta_2 = theta[1]
    return _log_probability(theta_1,theta_2,x)


def _log_lik(theta_1, theta_2, x):
    lik = 0.5 * norm.pdf(x, theta_1, SIGMA_x) + 0.5 * norm.pdf(x, theta_1 + theta_2, SIGMA_x)
    log_lik = np.sum(np.log(lik))
    return log_lik


def _log_probability(theta_1,theta_2,x):
    log_lik = _log_lik(theta_1, theta_2, x)

    log_prior = np.log(norm.pdf(theta_1,0, SIGMA_1)) + np.log(norm.pdf(theta_2,0, SIGMA_2))

    return log_lik+log_prior

def gen_X(n):
    res = []
    true_theta_1 = 0
    true_theta_2 = 1.
    for _ in range(n):
        coin  = np.random.rand()
        if coin < 0.5:
            add = np.random.randn()*SIGMA_x+true_theta_1
        else:
            add = np.random.randn()*SIGMA_x+true_theta_1+true_theta_2

        res.append(add)

    return np.array(res)


