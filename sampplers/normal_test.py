from sampplers.MetropolisHastings import metropolis_hastings
from test.stationary_distribution import MeanEmbeddingConsistanceTest

__author__ = 'kcx'
import autograd.numpy as np



def log_normal(x):
    return -np.dot(x,x)/2


thining_jump = 20
chain_size = 10000
results = np.zeros((thining_jump,3))

for thining in range(1,thining_jump,2):
    print('thining ', thining)
    pval = []

    for i in range(1000):
        x= metropolis_hastings(log_normal,N=chain_size,thining=thining)

        me = MeanEmbeddingConsistanceTest(x,log_normal)

        pval.append(me.compute_pvalue())


    res = np.percentile(pval, [5,10,15])*100.0
    results[thining] = res

print(results)

np.save('temp_quantiles.npy',results)