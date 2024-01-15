#verify that if \mu_SN^{*} is the best way to estimate \mu(\theta)

import datetime
import argparse
from itertools import product, permutations
import random as rd
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description=(
    "Compare different estimators for a Scenario,"
    " and save results at ./archive/scenario/VM.npy."
    " VM.npy contains a matrix of shape (9,16,3)."
    " r1 ranges from 0.1 to 0.5 with step of 0.05 (9 different values in total)."
    " r2 ranges from 0.25 to 0.4 with step of 0.25 (16 different values in total)."
    " VM[:,:,0],VM[:,:,1],VM[:,:,2] are the results of estimator_star, estimator_dagger"
    " and estimator_circ respectively."
    " Now we support 4 scenarios:"
    " SATenstein-QCP, clasp-weighted_sequence,"
    " LKH-uniform-1000, LKH-uniform-400"),
                                 formatter_class=argparse.
                                 ArgumentDefaultsHelpFormatter)
parser.add_argument("scenario", help="name of the scenario")
parser.add_argument("--max_parallism",
                    help=("#threads used to do the compuation."
                          " The larger, the better"),
                    type=int,
                    default=1)
args = parser.parse_args()

# clasp-weighted_sequence, LKH-uniform-1000, LKH-uniform-400 SATenstein-QCP
scenario = args.scenario
par = args.max_parallism

with open('./archive/%s/PM.npy' % scenario, 'rb') as f:
    perMatrix = np.load(f)
# handle empty entry in perMatrix
# This only happes once or twice in 10000000
entryIndice = np.argwhere(perMatrix['runtime'] == 0)
l = entryIndice.shape[0]
for i in range(l):
    entry = entryIndice[i, :]
    [configNum, insNum, seedNum] = entry
    perMatrix[configNum, insNum, seedNum]['runtime'] = np.sum(
        perMatrix['runtime'][configNum, insNum, :]) / np.sum(
            perMatrix['runtime'][configNum, insNum, :] != 0)

runtimeMatrix = perMatrix['runtime']
configNum, insNum, reTimes = runtimeMatrix.shape

# K = K_ratio * insNum with K_ratio \in [K_ratio_min, K_ratio_max] with stepsize K_ratio_step
# N = N_ratio * K

K_ratio_min = 0.1
# K_ratio_min = 0.3
K_ratio_max = 0.5
# K_ratio_max = 0.3
K_ratio_step = 0.05
K_num = int((K_ratio_max - K_ratio_min) / K_ratio_step) + 1

N_ratio_min = 0.25
# N_ratio_min = 1.25
N_ratio_max = 4
# N_ratio_max = 1.5
N_ratio_step = 0.25
N_num = int((N_ratio_max - N_ratio_min) / N_ratio_step) + 1

# n1: repeat times split training/test set
n1 = 2500
# n2: repeat times of evaluating \theta on K instances with N runs
n2 = 1

# the results
variationMatrix = np.zeros((K_num, N_num, 3))

insPool = set(range(insNum))

# for efficiency's sake
allscansequence = list(permutations(range(reTimes)))
seqLen = len(allscansequence)

def estimator_mu_star(config_index, KK, NN, instSample):
    # the universal best estimator
    a = int(NN) // int(KK)
    b = int(NN) % int(KK)

    lucky_ones = rd.sample(instSample, b)
    sumValue = 0.0

    for ins in instSample:
        Times = a
        if ins in lucky_ones:
            Times = a + 1
        scansequence = allscansequence[rd.randint(0, seqLen-1)]
        for iii in range(Times):
            sumValue += runtimeMatrix[config_index, ins, scansequence[iii]]
    return sumValue / NN


def estimator_mu_dagger(config_index, _, NN, instSample):
    # use most evaluations for selected instances
    a = int(NN) // reTimes
    b = int(NN) % reTimes
    sumValue = 0.0
    if b != 0:
        lucky_ones = rd.sample(instSample, a+1)
        for ins in lucky_ones[:-1]:
            sumValue += np.sum(runtimeMatrix[config_index, ins, :])
        choices = rd.sample(range(reTimes), b)
        sumValue += np.sum(runtimeMatrix[config_index, lucky_ones[-1], choices])
    else:
        lucky_ones = rd.sample(instSample, a)
        for ins in lucky_ones:
            sumValue += np.sum(runtimeMatrix[config_index, ins, :])

    return sumValue / NN


def estimator_mu_circ(config_index, _, NN, instSample):
    # every time randomly select an instance to evaluate
    lucky_ones = rd.sample(list(product(instSample, range(reTimes))), NN)
    sumValue = 0.0
    for one in lucky_ones:
        sumValue += runtimeMatrix[config_index, one[0], one[1]]
    return sumValue / NN

# def simulate_on_N_K(item):
#     # 1. for each config, sample training instances and test instances for n1 times
#     # 2. estimate its cost for n2 times
#     # 3. average on all configs
#     # for item in product(range(K_num), range(N_num)):
#     KK = int(np.ceil(insNum * (K_ratio_min + item[0] * K_ratio_step)))
#     NN = int(np.ceil(KK * (N_ratio_min + item[1] * N_ratio_step)))

#     accu_var_star = 0.0
#     accu_var_dagger = 0.0
#     accu_var_circ = 0.0
#     for ll in range(5):
#         # call estimator to estimate this config's cost
#         # based on different K instances and different runs (seeds)
#         print("(%d,%d, #config: %d)" % (item[0], item[1], ll))
#         # true_cost = trueCost[ll]
#         accu_config_star = 0.0
#         accu_config_dagger = 0.0
#         accu_config_circ = 0.0
#         for _ in range(n1):
#             # sample K training instances from all instances
#             # sample insNum/2 test instances from left instances
#             trainSample = rd.sample(insPool, KK)
#             testSample = rd.sample(insPool - set(trainSample), insNum // 2)
#             true_cost = np.mean(runtimeMatrix[ll, testSample, :])
#             for _ in range(n2):
#                 accu_config_star += (estimator_mu_star(ll, KK, NN, trainSample) - true_cost) ** 2
#                 accu_config_dagger += (estimator_mu_dagger(ll, KK, NN, trainSample) - true_cost) ** 2
#                 accu_config_circ += (estimator_mu_circ(ll, KK, NN, trainSample) - true_cost) ** 2

#         # ex_true_cost = true_cost / n1
#         accu_var_star += accu_config_star / (n1 * n2)
#         # print(accu_var_star)
#         accu_var_dagger += accu_config_dagger / (n1 * n2)
#         # print(accu_var_dagger)
#         accu_var_circ += accu_config_circ / (n1 * n2)
#         # print(accu_var_circ)

#     mean_var_star = accu_var_star / configNum
#     mean_var_dagger = accu_var_dagger / configNum
#     mean_var_circ = accu_var_circ / configNum
#     return item[0], item[1], mean_var_star, mean_var_dagger, mean_var_circ

def simulate_on_N_K_new(item):
    # 1. sample training instances and test instances for n1 times
    # 2. for each config, estimate its cost for n2 times
    # 3. average on all different splits of training/test samples
    # for item in product(range(K_num), range(N_num)):
    KK = int(np.ceil(insNum * (K_ratio_min + item[0] * K_ratio_step)))
    NN = int(np.ceil(KK * (N_ratio_min + item[1] * N_ratio_step)))

    accu_var_star = 0.0
    accu_var_dagger = 0.0
    accu_var_circ = 0.0
    for nn in range(n1):
        # sample K training instances from all instances
        # sample insNum/2 test instances from left instances
        print("(%d,%d, #split of train/test: %d)" % (item[0], item[1], nn))
        trainSample = rd.sample(insPool, KK)
        testSample = rd.sample(insPool - set(trainSample), insNum // 2)

        accu_config_star = 0.0
        accu_config_dagger = 0.0
        accu_config_circ = 0.0
        for ll in range(5):
            # call estimator to estimate this config's cost
            true_cost = np.mean(runtimeMatrix[ll, testSample, :])
            for _ in range(n2):
                # n different runs on K instances
                accu_config_star += (estimator_mu_star(ll, KK, NN, trainSample) -
                                     true_cost)**2
                accu_config_dagger += (
                    estimator_mu_dagger(ll, KK, NN, trainSample) - true_cost)**2
                accu_config_circ += (
                    estimator_mu_circ(ll, KK, NN, trainSample) - true_cost)**2

        # ex_true_cost = true_cost / n1
        accu_var_star += accu_config_star / (configNum * n2)
        # print(accu_var_star)
        accu_var_dagger += accu_config_dagger / (configNum * n2)
        # print(accu_var_dagger)
        accu_var_circ += accu_config_circ / (configNum * n2)
        # print(accu_var_circ)

    mean_var_star = accu_var_star / n1
    mean_var_dagger = accu_var_dagger / n1
    mean_var_circ = accu_var_circ / n1
    return item[0], item[1], mean_var_star, mean_var_dagger, mean_var_circ


r = Parallel(n_jobs=par)(delayed(simulate_on_N_K_new)(item)\
                         for item in product(range(K_num), range(N_num)))
KK_list, NN_list, res_star, res_dagger, res_circ = zip(*r)
for ii, _ in enumerate(KK_list):
    variationMatrix[KK_list[ii], NN_list[ii], 0] = res_star[ii]
    variationMatrix[KK_list[ii], NN_list[ii], 1] = res_dagger[ii]
    variationMatrix[KK_list[ii], NN_list[ii], 2] = res_circ[ii]

outputDir = "./archive/%s/" % scenario
np.save(outputDir + 'VM', variationMatrix)
print(datetime.datetime.now())
