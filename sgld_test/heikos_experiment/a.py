import numpy as np

p_vals = np.load('pvals.npy')
evals  = np.load('no_evals.npy')

p_vals = np.min(p_vals,axis=1)

print(p_vals)
epsilon = np.logspace(-4,0,50)

import matplotlib.pyplot as plt
#
# plt.plot(evals,p_vals)
# plt.show()
#
#
#
plt.plot(epsilon,p_vals)
plt.show()


import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(epsilon, evals, p_vals)
ax.legend()
ax.set_xlabel('epsilon')
ax.set_ylabel('likelihood evals')
ax.set_zlabel('p-value')

plt.show()
fig.savefig('../../write_up/img/heikos_experiment.pdf')