import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


fname = "increasing_features_fixed_test.txt"

fields = ['p_value']
field_plot_names = {
                    'p_value': 'p-value',
                    'm': r'$m$'
                    }
def kwargs_gen(**kwargs):
    return kwargs

conditions = kwargs_gen(
                          D=1,
                          N_test=500,
                          N_fit=50000,
                          num_bootstrap=200,
                          sigma=1,
                          lmbda=0.01,
                        )

# x-axis of plot
x_field = 'm'

df = pd.read_csv(fname, index_col=0)

for field in fields:
    plt.figure()
    
    # filter out desired entries
    mask = (df[field] == df[field])
    for k,v in conditions.items():
        mask &= (df[k] == v)
    current = df.loc[mask]

    sns.boxplot(x=x_field, y=field, data=current.sort(x_field))

    plt.xlabel(field_plot_names[x_field])
    plt.ylabel(field_plot_names[field])

    fname_base = os.path.splitext(fname)[0]
    plt.savefig(fname_base + ".png")
    plt.savefig(fname_base + ".eps")
    
    # print info on number of trials
    print(field)
    print("Average number of trials: %d" % int(np.round(current.groupby(x_field).apply(len).mean())))
    print(current.groupby(x_field).apply(len))
    
plt.show()
