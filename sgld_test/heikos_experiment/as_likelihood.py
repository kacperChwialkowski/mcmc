__author__ = 'kcx'
import numpy as np


LIK_EVALUATIONS = 10.0**5


SGLD_BATCH_SIZE = 20.0
sgld_chain_size = int(LIK_EVALUATIONS/SGLD_BATCH_SIZE)
b = 2.31
a = 0.01584
epsilons = a*(b+np.arange())**(-0.55)
min_epsilons = epsilons


