from sgld_test.test import gen_X, log_probability, _log_probability

__author__ = 'kcx'
from scipy.stats import norm
import numpy as np
import seaborn as sns; sns.set(color_codes=True)

X = gen_X(5000)

a1= np.linspace(0,1,20)
a2= np.linspace(1.0,-1,20)

res = []

import  matplotlib.pyplot as plt

for theta in zip(a1,a2):
    res.append(log_probability(theta,X))

plt.plot(res)
plt.show()


# print(a)


#
# ax = sns.kdeplot(X)
# sns.plt.show()
# exit(0)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
theta1 = np.arange(-4, 4 ,0.25)
the_len = len(theta1)
theta2 = np.arange(-4, 4, 0.25)
theta1, theta2 = np.meshgrid(theta1, theta2)
Z = np.copy(theta1)

for i in range(the_len):
    for j in range(the_len):
        Z[i,j] = _log_probability(theta1[i,j],theta2[i,j],X)


fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(theta1, theta2, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
