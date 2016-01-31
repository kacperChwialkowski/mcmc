import numpy as np
from pandas import DataFrame
import seaborn as sns
from tools.latex_plot_init import plt

p_values = np.load('pvals.npy')

likelihood_evaluations = np.load('no_evals.npy') / 10 ** 6
epsilon = np.linspace(0.001, 0.2, 25)
epsilon = ["%.2f" % v for v in epsilon]
epsilon[0] = "0.001"



def to_data_frame(arr,epsilon):
    data = []
    for i, r in enumerate(arr):
        for eval in r:
            data.append([epsilon[i], eval])
    return DataFrame(data)

p_values = to_data_frame(p_values[::5], epsilon[::5])

# likelihood_evaluations = to_data_frame(likelihood_evaluations, epsilon)

plt.figure()
sns.boxplot(x=0, y=1, data=p_values, palette="BuGn_d")
plt.ylabel("p values")
plt.xlabel("epsilon")
plt.tight_layout()
plt.savefig('../../write_up/img/Heiko1.pdf')

plt.figure()
plt.plot(epsilon[::2],np.mean(likelihood_evaluations[::2],axis=1),'g')
plt.ylabel("likelihood evaluations")
plt.xlabel("epsilon")
plt.tight_layout()
plt.savefig('../../write_up/img/Heiko2.pdf')
#
# f, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
#
# sns.boxplot(x=0, y=1, data=p_values, palette="BuGn_d", ax=ax1)
# ax1.set_ylabel("p values")
# ax1.set_xlabel("")
#
# y2 = likelihood_evaluations
# sns.barplot(x=0, y=1, data=likelihood_evaluations, palette="RdBu_d", ax=ax2)
#
# ax2.set_ylabel("Likelihood evaluations")
#
# sns.despine(bottom=True)
# ax2.set(xlabel='epsilon')
#
# f.tight_layout()
# plt.show()

# f.savefig('../../write_up/img/Heiko.pdf')
