__author__ = 'kcx'

import numpy as np
n_samples=300
# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([0, 0])

# generate zero centered stretched Gaussian data
C = np.array([[2, 0], [0, 2]])
# stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
# points = np.vstack([shifted_gaussian, stretched_gaussian])

points = shifted_gaussian
import matplotlib.pyplot as plt
plt.scatter(x=points[:,0], s=35,y=points[:,1], alpha=0.5,edgecolors='none')



plt.axis('off')
plt.savefig('../write_up/presentation/img/mixtureOfNormal.pdf')

plt.show()
# import seaborn as sns; sns.set(color_codes=True)
# sns.set_style("whitegrid")
# sns.despine(left=True)
#
# sns_plot = sns.regplot(x=points[:,0], y=points[:,1],fit_reg=False)
# sns.plt.show()
#
#
# fig = sns_plot.get_figure()
# fig.savefig('../write_up/presentation/img/mixtureOfNormal.pdf')