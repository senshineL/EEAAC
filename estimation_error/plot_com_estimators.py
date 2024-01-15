# for generating raw data file (color scale pictures)
import numpy as np
import matplotlib.pyplot as plt

# clasp-weighted_sequence, LKH-uniform-1000, LKH-uniform-400 SATenstein-QCP
scenarios = [
    'SATenstein-QCP', 'clasp-weighted_sequence', 'LKH-uniform-400',
    'LKH-uniform-1000'
]
K_values = [500, 120, 250, 250]

r1_values = []
min_value = 0.1
step = 0.05
tmp = min_value
for _ in range(9):
    r1_values.append(tmp)
    tmp += step

r2_values = []
min_value = 0.25
step = 0.25
tmp = min_value
for _ in range(16):
    r2_values.append(tmp)
    tmp += step

for i, r1 in enumerate(r1_values):
    for j, scenario in enumerate(scenarios):
        K = K_values[j]
        with open('./archive/%s/VM.npy' % scenario, 'rb') as f:
            vm = np.load(f)
        plot_values = vm[i, :, :]
        x_list = []
        for r2 in r2_values:
            x_list.append(int(K * r2))

        # plt.margins(x=0.015)
        # plt.margins(y=0.015)
        plt.rcParams['figure.figsize'] = 8, 6
        plt.rcParams.update({'font.size': 20})
        f, ax = plt.subplots()
        plt.xlabel("N")
        plt.ylabel("mean estimation error by PAR-10")
        ax.plot(x_list, plot_values[:, 0], '-', linewidth=3, label='$\hat{u}_{S_{N}^{\\ast}}(\\theta)$')
        ax.plot(x_list, plot_values[:, 1], ':', linewidth=3, label='$\hat{u}_{S_{N}^{\dagger}}(\\theta)$')
        ax.plot(x_list, plot_values[:, 2], '--', linewidth=3, label='$\hat{u}_{S_{N}^{\circ}}(\\theta)$')
        legend = ax.legend(loc='upper right', prop={'size': 25})
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        # plt.colorbar(sc)
        [i.set_linewidth(2) for i in ax.spines.values()]
        plt.tight_layout()
        plt.savefig("%s_%i_com_estimator.pdf" % (scenario, i))
        # plt.savefig("%s_%i_com_estimator.pdf" % (scenario, i),
        # bbox_inches='tight')
        plt.clf()
