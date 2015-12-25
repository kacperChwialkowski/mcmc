from sampplers.MetropolisHastings import metropolis_hastings
import numpy as np
__author__ = 'kcx'


def log_normal(x):
    return -np.dot(x,x)/2


x= metropolis_hastings(log_normal,N=1000,thining=3)



