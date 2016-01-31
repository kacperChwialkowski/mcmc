import numpy as np
from pandas import DataFrame
import seaborn as sns
from tools.latex_plot_init import plt


p_vals = np.load('pvals.npy')
evals  = np.load('no_evals.npy')/10**6
epsilon = np.linspace(0.001, 0.2,25)

epsilon = ["%.2f" % v for v in epsilon]
epsilon[0]="0.001"

data_pvals = []

for i,r in enumerate(p_vals):
    for eval in r:
         data_pvals.append( [epsilon[i], eval])

data_pvals = DataFrame(data_pvals)

data_evals = []
for i,r in enumerate(evals):
    for eval in r:
        data_evals.append([epsilon[i], eval])


data_evals = DataFrame(data_evals)


f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

sns.boxplot(x=0,y=1,data = data_pvals, palette="BuGn_d", ax=ax1)
ax1.set_ylabel("p values")
ax1.set_xlabel("")

y2 = evals
sns.barplot(x=0,y=1,data = data_evals,palette="RdBu_r", ax=ax2)

ax2.set_ylabel("Likelihood evaluations")

sns.despine(bottom=True)
ax2.set(xlabel='epsilon')


f.tight_layout()
plt.show()

f.savefig('../../write_up/img/Heiko.pdf')