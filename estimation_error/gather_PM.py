# Gather performance matrix
import time
import subprocess
import random
import sys
import os
import datetime
import argparse
import numpy as np

def extract_result(name_list, ones):
    for name in name_list:
        print(name)
        if 'Config' in name:
            configNum = int(name[name.find('g')+1 : name.find('I')-1])
            insNum = int(name[name.find('s')+1 : name.find('S')-1])
            rngseed = int(name[name.find('d')+1:])
            with open(outputDir + name, 'r') as fp:
                lines = fp.read().strip().split('\n')
                flag = False
                for line in lines:
                    if 'Result for' not in line:
                        continue
                    flag = True
                    values = line[line.find(\
                            ':') + 1:].strip().replace(' ', '').split(',')
                    (result, runtime, _, quality, rngseed) = values[0:5]
                    runtime = float(runtime)
                    quality = float(quality)
                    rngseed = int(rngseed)
                    if result in ['TIMEOUT', 'CRASHED']:
                        runtime = capTime * punish
                if not flag:
                    print("Not found result for SMAC/PARAMILS in %s" % (name))
                    ones.append((configNum, insNum, rngseed))
                    continue
            for iq in range(reTimes):
                if performanceMatrix[configNum, insNum, iq]['runtime'] == 0:
                    performanceMatrix[configNum, insNum, iq] = (result, runtime,\
                                                                quality, rngseed)
                    break

parser = argparse.ArgumentParser(
    description=("Gather Performance Matrix for a Scenario,"
                 " and save results at ./archive/scenario/PM.npy."
                 " PM.npy contains a matrix of shape (#config, #insts, 5)."
                 " Now we support 4 scenarios:"
                 " SATenstein-QCP, clasp-weighted_sequence,"
                 " LKH-uniform-1000, LKH-uniform-400"),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("scenario", help="name of the scenario")
parser.add_argument("--max_parallism", help=("#threads used to gather the performance"
                                             " matrix. The larger, the better"), type=int,
                    default=1)
args = parser.parse_args()

# use current system time to set random state
random.seed()
os.chdir('../')
maxParalism = args.max_parallism
scenario = args.scenario

if scenario == 'SATenstein-QCP':
    configNumber = 500
    insNumber = 500
    reTimes = 5
    capTime = 5
    punish = 10
    configFile = "./estimation_error/configs/SATenstein-QCP.txt"
    algo = ("python -u ./target_algorithms/sat/satenstein/wrapper.py "
            "--mem-limit 3000 --sat-checker ./target_algorithms/sat/tools/SAT ")
    insFile = "./estimation_error/instance_index/SATenstein-QCP.txt"
elif scenario == 'CPLEX-REGIONS100':
    configNumber = 1000
    insNumber = 2000
    reTimes = 1
    capTime = 5
    punish = 10
    configFile = "./estimation_error/configs/CPLEX-REGIONS100.txt"
    algo = ("python ./target_algorithms/mip/cplex12.6/wrapper.py "\
            "--mem-limit 1024 --obj-file ./instances/mip/sets/Regions100/solubility.txt ")
    insFile = "./estimation_error/instance_index/CPLEX-REGIONS100.txt"
elif scenario == 'lpg-depots':
    configNumber = 500
    insNumber = 250
    reTimes = 5
    capTime = 10
    punish = 10
    configFile = "./estimation_error/configs/lpg-depots.txt"
    algo = ("python ./target_algorithms/planning/lpg/wrapper.py --mem-limit 3000"
            " --solution-validator ./target_algorithms/planning/VAL/validate "
            " --problem-enc ./instances/planning/data/depots/depots.pddl ")
    insFile = "./estimation_error/instance_index/lpg-depots.txt"
elif scenario == 'clasp-weighted_sequence':
    configNumber = 500
    insNumber = 120
    reTimes = 5
    capTime = 25
    punish = 10
    configFile = "./estimation_error/configs/clasp-weighted_sequence.txt"
    algo = ("python -u ./target_algorithms/asp/clasp-3.1.4/wrapper.py --mem-limit 2048 ")
    insFile = "./estimation_error/instance_index/clasp-weighted_sequence.txt"
elif scenario == 'LKH-uniform-1000':
    configNumber = 500
    insNumber = 250
    reTimes = 5
    capTime = 10
    punish = 10
    configFile = "./estimation_error/configs/LKH-uniform-1000.txt"
    algo = ("python -u ./target_algorithms/tsp/LKH/wrapper.py --mem-limit 1024"
            " --solutionity ./instances/tsp/sets/tsp-1000/optimum.json ")
    insFile = "./estimation_error/instance_index/LKH-uniform.txt"
elif scenario == 'LKH-uniform-400':
    configNumber = 500
    insNumber = 250
    reTimes = 5
    capTime = 10
    punish = 10
    configFile = "./estimation_error/configs/LKH-uniform-400.txt"
    algo = ("python -u ./target_algorithms/tsp/LKH/wrapper.py --mem-limit 1024"
            " --solutionity ./instances/tsp/sets/tsp-400/optimum.json ")
    insFile = "./estimation_error/instance_index/LKH-uniform-400.txt"
else:
    print("Unrecognized scenario:%s" % scenario)
    sys.exit(1)

outputDir = "./estimation_error/results/%s/" % scenario
final_outputDir = "./estimation_error/archive/%s/" % scenario

performanceMatrix = np.zeros((configNumber, insNumber, reTimes),
                             dtype=[
                                 ('result', 'U10'),
                                 ('runtime', 'f8'),
                                 ('quality', 'f16'),
                                 ('seed', 'i8'),
                             ])

with open(configFile, 'r') as f:
    configs = f.read().strip().split('\n')
with open(insFile, 'r') as f:
    insts = f.read().strip().split('\n')

runningTask = 0
processSet = set()
for i in range(configNumber):
    for j in range(insNumber):
        seedSet = set()
        for k in range(reTimes):
            while True:
                if runningTask >= maxParalism:
                    time.sleep(0.1)
                    finished = [
                        pid for pid in processSet if pid.poll() is not None
                    ]
                    processSet -= set(finished)
                    runningTask = len(processSet)
                    continue
                else:
                    config = configs[i]
                    instance = insts[j]
                    seed = random.randint(0, 1000000)
                    while seed in seedSet:
                        seed = random.randint(0, 1000000)
                    output_file = '%sConfig%d_Ins%d_Seed%d' % (outputDir, i, j, seed)
                    cmd = ("%s %s %d %.1f %d %d %s > %s" %\
                        (algo, instance, 0, capTime, 0, seed, config, output_file))
                    print(cmd)
                    # sys.exit(0)
                    processSet.add(subprocess.Popen(cmd, shell=True))
                    runningTask = len(processSet)
                    break

# check if subprocess all exits
while processSet:
    time.sleep(5)
    print('Still %d sub process not exits' % len(processSet))
    finished = [pid for pid in processSet if pid.poll() is not None]
    processSet -= set(finished)

nameList = os.listdir(outputDir)
need_test_ones = []
extract_result(nameList, need_test_ones)

repeat = 1
while need_test_ones:
    print("Repeat %d test, %d ones no result in there\n" % (repeat, len(need_test_ones)))
    nameList = []
    runningTask = 0
    processSet = set()
    total = len(need_test_ones)
    for i, one in enumerate(need_test_ones):
        nameList.append('Config%d_Ins%d_Seed%d' % (one[0], one[1], one[2]))
        while True:
            if runningTask >= maxParalism:
                time.sleep(0.1)
                finished = [
                    pid for pid in processSet if pid.poll() is not None
                ]
                processSet -= set(finished)
                runningTask = len(processSet)
                continue
            else:
                config = configs[one[0]]
                instance = insts[one[1]]
                seed = one[2]
                output_file = '%sConfig%d_Ins%d_Seed%d' % (outputDir, one[0], one[1], seed)
                cmd = ("%s %s %d %.1f %d %d %s > %s" %\
                        (algo, instance, 0, capTime, 0, seed, config, output_file))
                print(("Repeat: %d, %d/%d, cmd:\n" % (repeat, i+1, total)) + cmd)
                processSet.add(subprocess.Popen(cmd, shell=True))
                runningTask = len(processSet)
                break

    # check if subprocess all exits
    while processSet:
        time.sleep(5)
        print('Still %d sub process not exits' % len(processSet))
        finished = [pid for pid in processSet if pid.poll() is not None]
        processSet -= set(finished)

    need_test_ones = []
    extract_result(nameList, need_test_ones)
    repeat += 1

np.save(outputDir + 'PM', performanceMatrix)
np.save(final_outputDir + 'PM', performanceMatrix)
print(datetime.datetime.now())
