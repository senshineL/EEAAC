# We demonstrate the relationship between estimation error and m

import datetime
import argparse
import random as rd
from itertools import permutations
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description=(
    " Estimation error for different m for a Scenario,"
    " and save results at ./archive/scenario/VM_m.npy."
    " VM_m.npy contains a matrix of shape (n,3), where"
    " n is the considered values of m."
    " VM_m[:,0],VM_m[:,1],VM_m[:,2] are the results of uniform estimation error,"
    " estimation error for config^star (with the best training performance) and"
    " the training performance of config^star respectively."
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
# dont know whether should we operate on log10 scale
# log10_runtimeMatrix_config_ins = np.log10(runtimeMatrix)
configNum, insNum, reTimes = runtimeMatrix.shape
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
        scansequence = allscansequence[rd.randint(0, seqLen - 1)]
        for iii in range(Times):
            sumValue += runtimeMatrix[config_index, ins, scansequence[iii]]
    return sumValue / NN

# diviation for different m
# fix K as half training instances
# fix N as K * reTimes
K_ratio = 0.5
K = int(insNum * K_ratio)

N_ratio = reTimes
N = int(K * N_ratio)

# split training/test sets for n1 times
n1 = 2500
# estimate a config's performance for n3 times on a given training set
n3 = 1

# consider uniform deviation and the deviation for the config with best training cost
considered_m = list(range(1, configNum+1))
deviationMatrix_m = np.zeros((configNum, 3))


def simulate_on_split_for_all_m(sample_item):
    # 1. sample training/test set for n1 times
    # 2. based on each set split, range up a config set
    # calculate uniform deviation and
    # training deviation for each m
    # 3. average over all splits
    nn, train_sample, test_sample = sample_item
    print("Deviation for m, #split of train/test: %d)" % (nn))

    configPool = set(range(configNum))
    config_list = []
    cost_record_list = [] # for each config considered
    deviation_record_list = [] # for each config considered

    for m in considered_m:
        new_configs = rd.sample(configPool, m-len(config_list))
        configPool -= set(new_configs)
        config_list.extend(new_configs)

        for config in new_configs:
            # find deviation and training performance of this config
            true_cost = np.mean(runtimeMatrix[config, test_sample, :])
            train_cost = None
            config_deviation = None

            for _ in range(n3):
                # estimste this config's performance
                estimated_cost = estimator_mu_star(config, K, N, train_sample)
                deviation = true_cost - estimated_cost
                if train_cost is None or estimated_cost < train_cost:
                    train_cost = estimated_cost
                    config_deviation = deviation

            cost_record_list.append(train_cost)
            deviation_record_list.append(config_deviation)
    return cost_record_list, deviation_record_list

sample_list = []
for nnn in range(n1):
    trainSample = rd.sample(insPool, K)
    testSample = rd.sample(insPool - set(trainSample), insNum // 2)
    sample = []
    sample_list.append((nnn, trainSample, testSample))
r = Parallel(n_jobs=par)(delayed(simulate_on_split_for_all_m)(item) for item in sample_list)

cost_list_list, deviation_list_list = zip(*r)
cost_matrix = np.zeros((len(cost_list_list), len(cost_list_list[0])))
deviation_matrix = np.zeros((len(deviation_list_list), len(deviation_list_list[0])))

for item, _ in enumerate(cost_list_list):
    cost_matrix[item, :] = cost_list_list[item]
    deviation_matrix[item, :] = deviation_list_list[item]

outputDir = "./archive/%s/" % scenario
# np.save(outputDir + 'cost_matrix_m', cost_matrix)
# np.save(outputDir + 'deviation_matrix_m', deviation_matrix)
for i in range(cost_matrix.shape[1]):
    deviationMatrix_m[i, 0] = np.mean(np.max(deviation_matrix[:, 0:i+1], axis=1))
    minimum_index = np.argmin(cost_matrix[:, 0:i+1], axis=1)
    tmp1 = 0.0
    tmp2 = 0.0
    for ii, index in enumerate(minimum_index):
        tmp1 += deviation_matrix[ii, index]
        tmp2 += cost_matrix[ii, index]
    deviationMatrix_m[i, 1] = tmp1 / n1
    deviationMatrix_m[i, 2] = tmp2 / n1

np.save(outputDir + 'DM_m', deviationMatrix_m)
print(datetime.datetime.now())
