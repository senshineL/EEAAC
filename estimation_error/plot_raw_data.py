# for generating raw data file (color scale pictures)
import numpy as np
import matplotlib.pyplot as plt

# clasp-weighted_sequence, LKH-uniform-1000, LKH-uniform-400 SATenstein-QCP
scenarios = ['SATenstein-QCP', 'clasp-weighted_sequence', 'LKH-uniform-400', 'LKH-uniform-1000']
for scenario in scenarios:
    with open('./archive/%s/PM.npy' % scenario, 'rb') as f:
        perMatrix = np.load(f)

    # handle empty entry in perMatrix
    # This only happes once or twice in 10000000
    entryIndice = np.argwhere(perMatrix['runtime'] == 0)
    l = entryIndice.shape[0]
    for i in range(l):
        entry = entryIndice[i, :]
        [configNum, insNum, seedNum] = entry
        perMatrix[configNum, insNum, seedNum]['runtime'] = np.sum(perMatrix['runtime'][configNum, insNum, :]) / np.sum(perMatrix['runtime'][configNum, insNum, :] != 0)

    # draw gray-color graphs on log-10 scale runtime
    runtimeMatrix = perMatrix['runtime']
    runtimeMatrix_config_ins = np.mean(runtimeMatrix, axis=2)
    log10_runtimeMatrix_config_ins = np.log10(runtimeMatrix_config_ins)
    # sort log_runtimeMatrix_config_ins at two dimwnsions
    a = np.mean(runtimeMatrix_config_ins, axis=1)
    b = np.mean(runtimeMatrix_config_ins, axis=0)

    tmp_a = np.argsort(a)
    tmp_aa = log10_runtimeMatrix_config_ins[tmp_a, :]
    tmp_b = np.argsort(b)
    tmp_bb = tmp_aa[:, tmp_b]

    log10_minValue = np.ma.min(tmp_bb)
    log10_maxValue = np.ma.max(tmp_bb)
    log10_scale = log10_maxValue - log10_minValue

    cmx = plt.get_cmap('Greys')
    cmx_r = plt.get_cmap('Greys_r')

    # tmp_bb_colorValue = (tmp_bb - log10_minValue) / log10_scale + log10_minValue
    tmp_bb_colorValue = tmp_bb

    xList = []
    yList = []
    cList = []
    for i in range(tmp_bb_colorValue.shape[1]):
        xList.extend([i+1] * tmp_bb_colorValue.shape[0])
        yList.extend(list(range(1, tmp_bb_colorValue.shape[0] + 1)))
        cList.extend(list(tmp_bb_colorValue[:, i]))

    plt.rcParams.update({'font.size': 20})
    sc = plt.scatter(xList, yList, marker='.', c=cList, cmap=cmx_r)
    plt.margins(x=0.015)
    plt.margins(y=0.015)
    plt.xlabel("instances (sorted by hardness)")
    plt.ylabel("configurations (sorted by PAR-10)")
    plt.colorbar(sc)

    plt.savefig("%s_raw_data.png" % scenario, s=0.1, dpi=150, bbox_inches='tight')
    plt.clf()
