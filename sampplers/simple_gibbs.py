from multiprocessing.pool import Pool
import random
import math
import autograd.numpy as np
from autograd import grad



def log_ugly(x):
    return -(x/10.0 + np.sin(x) )**2.0/2.0

def log_ugly(x):
    return  -np.dot(x,x)/2

grad_log_ugly_ref = grad(log_ugly)

a = grad(log_ugly)


def test_function(x,omega=0.4):
    return np.exp( -(x-omega)**2.0/2.0)

grad_test = grad(test_function)

def stat(x):
    return grad_log_ugly_ref(x)*test_function(x) + grad_test(x)

x_prev = np.random.randn()*10



def log_ugly_miss(x):
    return -(x/9.97 + np.sin(x) )**2.0/2.0


grad_log_ugly_mis = grad(log_ugly)


def stat_miss(x):
    return log_ugly_miss(x)*test_function(x) + grad_test(x)




A = [x_prev]
thining = 15
N = 100000
for i in range(N*thining-1):
    guess = 2*np.random.randn()+x_prev
    old_log_lik = log_ugly(x_prev)
    new_log_lik = log_ugly(guess)
    if new_log_lik > old_log_lik:
        A.append(guess)
    else:
        u = random.uniform(0.0,1.0)
        if (u < math.exp(new_log_lik - old_log_lik)):
            A.append(guess)
        else:
            A.append(x_prev)
    x_prev = A[-1]

import matplotlib.pyplot as plt
# plt.plot(A)
# plt.show()


A= A[::thining]
print(len(A),N)



assert len(A)==N


pool = Pool(processes=4)              # start 4 worker processes


def get_cum(A,stat,pool):

    A = pool.map(stat, A)
    A = np.array(A)
    cum_mean = np.cumsum(A) / np.arange(N)
    cum_second_moment = np.cumsum(A ** 2.0) / np.arange(N)
    cum_std = np.sqrt(-(cum_mean) ** 2 + cum_second_moment)
    cum_stat = (np.sqrt(np.arange(N)) * cum_mean) / cum_std
    return cum_stat[::100], cum_std[::100]

cm,cstd = get_cum(A,stat,pool)


# cm_mis,_ = get_cum(A,stat_miss,pool)




plt.plot(cm,'k')

plt.plot(3*cstd,'b--')

plt.plot(-3*cstd,'b--')

# plt.plot(cm_mis,'r')




plt.show()
#
#
# rows = int(N/ window)
#
# A = np.reshape(A,(rows, window))
#
# imnormal = np.zeros(rows)
# for row in range(rows):
#     imnormal[row] = np.abs((np.sqrt(1000.0) * np.mean(A[row])) / np.var(A[row]))
#
#
#
# plt.plot(imnormal)
#
# plt.show()