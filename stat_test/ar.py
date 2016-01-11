from statsmodels.tsa.stattools import acf

__author__ = 'kcx'
import numpy as np

def simulate( nPeriod, nPath,beta):
    noise =  np.random.randn(nPeriod, nPath)
    sims = np.zeros((nPeriod, nPath))
    sims[0] = noise[0]
    sqrt_beta = np.sqrt(1 - beta ** 2)
    for period in range(1, nPeriod):
        sims[period] = beta*sims[period-1] + sqrt_beta *noise[period]
    return sims


if __name__ == "__main__":
    w = simulate(10000,1,0.9)
    print(acf(np.sign(w),nlags=3))
    print(np.dot(w.T,w)/1000)


    # import matplotlib.pyplot as plt
    #
    # plt.plot(w)
    # plt.show()
    #
    # w = simulate(1000,3,0.9)
    # print(acf(np.sign(w[:,1]),nlags=3))
    # print(np.dot(w.T,w)/1000)