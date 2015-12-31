from sgld_test.test import gen_X, log_probability, _log_probability

__author__ = 'kcx'
import numpy as np
import seaborn as sns; sns.set(color_codes=True)

N=200
X = gen_X(N)

import matplotlib.pyplot as plt
theta1 = np.arange(-2, 2 ,0.25)
the_len = len(theta1)
theta2 = np.arange(-2, 2, 0.25)
theta1, theta2 = np.meshgrid(theta1, theta2)
Z = np.copy(theta1)



log_den = np.exp(-(X**2/24.0)) /(4.0*np.sqrt(6.0*np.pi))+ np.exp(-(X**2/26))/(2*np.sqrt(26*np.pi))

log_den =  np.sum(np.log(log_den))

for i in range(the_len):
    for j in range(the_len):

        probability = _log_probability(theta1[i, j], theta2[i, j], X)
        Z[i,j] = np.exp(probability-log_den)


print(Z)
plt.figure()
CS = plt.contour(theta1, theta2, Z,10)


plt.show()
