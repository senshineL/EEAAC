# We demonstrate the relationship between estimation error and N

import datetime
import argparse
import random as rd
from itertools import permutations
import numpy as np
from joblib import Parallel, delayed

parser = argparse.ArgumentParser(description=(
    " Estimation error for different N for a Scenario,"
    " and save results at ./archive/scenario/VM_N.npy."
    " VM_N.npy contains a matrix of shape (n,3), where"
    " n is the considered values of N."
    " VM_N[:,0],VM_N[:,1],VM_N[:,2] are the results of uniform estimation error,"
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
configNum, insNum, reTimes = runtimeMatrix.shape
insPool = set(range(insNum))
# for efficiency's sake
allscansequence = list(permutations(range(reTimes)))
seqLen = len(allscansequence)

def estimator_mu_star_incremental(config_index, KK, NN, instSample):
    # the estimator with the smallest variance
    # return every estimated value from 1-> NN
    # like looping method
    # a = int(NN) // int(KK)
    # b = int(NN) % int(KK)

    # lucky_ones = rd.sample(instSample, b)
    times_record = []
    for _ in instSample:
        times_record.append(set(range(reTimes)))

    value_list = []
    sumValue = 0.0
    insIndexSet = set(range(KK))
    for estimate_times in range(1, NN+1):
        tmp = rd.sample(insIndexSet, 1)
        insIndex = tmp[0]
        insIndexSet.remove(insIndex)

        tmp = rd.sample(times_record[insIndex], 1)
        repe = tmp[0]
        times_record[insIndex].remove(repe)

        sumValue += runtimeMatrix[config_index, instSample[insIndex], repe]
        value_list.append(sumValue/estimate_times)
        # print("%d: " % estimate_times +str(value_list) + '\n')
        if not insIndexSet:
            insIndexSet = set(range(KK))
    return value_list

# deviation for different N
# fix m as configNum
# fix K as half instances
K_ratio = 0.5
K = int(insNum * K_ratio)

largestN = K * reTimes
considered_N = list(range(1, largestN + 1))
N_num = len(considered_N)
# split training/test sets for n1 times
n1 = 2500
# n1 = 1
# estimate a config's performance for n3 times on a given training set
n3 = 1
# consider uniform deviation and the deviation for the config with best training cost
deviationMatrix_N = np.zeros((N_num, 3))

def simulate_on_split_for_all_N(sample_item):
    # 1. sample training/test set for n1 times
    # 2. based on each set split, estimate each config's
    # performance based on N
    # range up N
    # calculate uniform deviation and
    # training deviation for each N
    # 3. average over all splits
    nn, train_sample, test_sample = sample_item
    print("Deviation for N, #split of train/test: %d)" % (nn))

    cost_record_list = np.zeros((configNum, N_num))  # for each N considered
    deviation_record_list = np.zeros((configNum, N_num))  # for each N considered
    config_list = range(configNum)
    for config in config_list:
        # find deviation and training performance of this config
        true_cost = np.mean(runtimeMatrix[config, test_sample, :])

        # estimste this config's performance
        cost_record_list[config, :] = estimator_mu_star_incremental(config, K,\
                                                                    largestN, train_sample)
        deviation_record_list[config, :] = true_cost - cost_record_list[config, :]

    return cost_record_list, deviation_record_list

sample_list = []
for nnn in range(n1):
    trainSample = rd.sample(insPool, K)
    testSample = rd.sample(insPool - set(trainSample), insNum // 2)
    sample_list.append((nnn, trainSample, testSample))
r = Parallel(n_jobs=par)(delayed(simulate_on_split_for_all_N)(item)
                         for item in sample_list)

cost_list_list, deviation_list_list = zip(*r)
deviation_matrix = np.zeros((configNum, N_num, n1))
deviation_deviation = np.zeros((N_num, n1))
cost_cost = np.zeros((N_num, n1))

for item, _ in enumerate(cost_list_list):
    sub_cost_matrix = cost_list_list[item]
    sub_deviation_matrix = deviation_list_list[item]

    minimum_index = np.argmin(sub_cost_matrix, axis=0)
    for i, index in enumerate(minimum_index):
        cost_cost[i, item] = sub_cost_matrix[index, i]
        deviation_deviation[i, item] = sub_deviation_matrix[index, i]

    deviation_matrix[:, :, item] = sub_deviation_matrix


deviationMatrix_N[:, 0] = np.mean(np.max(deviation_matrix, axis=0), axis=1)
deviationMatrix_N[:, 1] = np.mean(deviation_deviation, axis=1)
deviationMatrix_N[:, 2] = np.mean(cost_cost, axis=1)
outputDir = "./archive/%s/" % scenario
np.save(outputDir + 'DM_N', deviationMatrix_N)
print(datetime.datetime.now())
