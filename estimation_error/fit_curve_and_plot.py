import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# fit function for m
def func_m(x, a, b):
    return a*np.log(x) + b * np.sqrt(np.log(x))

# fit function for N
def func_N(x, a, b):
    return a + b * np.sqrt(np.divide(1.0, x))

# fit function for K
def func_K(x, a, b):
    return a * np.divide(1.0, x)+ b * np.sqrt(np.divide(1.0, x))

# clasp-weighted_sequence, LKH-uniform-1000, LKH-uniform-400 SATenstein-QCP
scenarios = [
    'SATenstein-QCP', 'clasp-weighted_sequence', 'LKH-uniform-400',
    'LKH-uniform-1000'
]

option = 'K'
if option == 'm':
    func = func_m
    ff = 'DM_m.npy'
    xdata = np.array(list(range(1, 501)))
elif option == 'N':
    func = func_N
    ff = 'DM_N.npy'
elif option == 'K':
    func = func_K
    ff = 'DM_K.npy'
else:
    print('Unrecognized option')
    sys.exit(1)
# 0: uniform estimation error
# 1: estimation error on config^{star}
option_1 = 1

for scenario in scenarios:
    with open('./archive/%s/%s' % (scenario, ff), 'rb') as f:
        dm_m = np.load(f)

    ydata = dm_m[:, option_1]
    xdata = list(range(1, ydata.shape[0]+1))

    # optimize parameters
    popt1, pcov1 = curve_fit(func, xdata, ydata)

    plt.margins(x=0.1)
    plt.margins(y=0.1)
    plt.rcParams['figure.figsize'] = 8, 6
    plt.rcParams.update({'font.size': 18})
    f, ax = plt.subplots()
    plt.xlabel(option)
    if option_1 == 0:
        plt.ylabel("uniform estimation error by PAR-10")
    elif option_1 == 1:
        plt.ylabel("estimation error on $\\theta^{\\ast}$ by PAR-10")
    ax.plot(xdata, func(xdata, *popt1), 'g--', linewidth=3, label='fit function')
    ax.plot(xdata, ydata, 'b-', linewidth=3,
            label='$\mathrm{%s\_es\_error(\Theta_{%s},%s)}$' %\
            ("uniform" if option_1 == 0 else "train",
             "train" if option == 'm' else "M",
             option))
    legend = ax.legend(loc='best', prop={'size': 20})
    ax.set_xlim(left=-5)
    ax.set_ylim(bottom=0)
    [i.set_linewidth(2) for i in ax.spines.values()]
    plt.tight_layout()
    if option_1 == 0 :
        output_file = "%s_DM_%s.pdf" % (scenario, option)
    elif option_1 == 1:
        output_file = "%s_DM_%s_train.pdf" % (scenario, option)
    plt.savefig(output_file)
    plt.clf()
