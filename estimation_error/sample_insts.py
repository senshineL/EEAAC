# sample instances and save them to /instance_index/
import random

scenario = 'LKH-uniform-400'
sourceFile1 = None
sourceFile2 = None

if scenario == 'SATenstein-QCP':
    insNumer = 2000
    insFile = "./instance_index/SATenstein-QCP.txt"
    sourceFile1 = "../instances/sat/sets/QCP/training.txt"
    sourceFile2 = "../instances/sat/sets/QCP/test.txt"
elif scenario == 'CPLEX-REGIONS100':
    insNumer = 2000
    insFile = "./instance_index/CPLEX-REGIONS100.txt"
    sourceFile1 = "../instances/mip/sets/Regions100/training.txt"
    sourceFile2 = "../instances/mip/sets/Regions100/test.txt"
elif scenario == 'lpg-depots':
    insNumer = 1000
    insFile = "./instance_index/lpg-depots.txt"
    sourceFile1 = "../instances/planning/sets/depots/training.txt"
    sourceFile2 = "../instances/planning/sets/depots/test.txt"
elif scenario == 'clasp-weighted_sequence':
    insNumer = 120
    insFile = "./instance_index/clasp-weighted_sequence.txt"
    sourceFile1 = "../instances/asp/sets/weighted-sequence/training.txt"
    sourceFile2 = "../instances/asp/sets/weighted-sequence/test.txt"
elif scenario == 'LKH-uniform-1000':
    insNumer = 500
    insFile = "./instance_index/LKH-uniform-1000.txt"
    sourceFile1 = "../instances/tsp/sets/tsp-1000/instance.txt"
elif scenario == 'LKH-uniform-400':
    insNumer = 250
    insFile = "./instance_index/LKH-uniform-400.txt"
    sourceFile1 = "../instances/tsp/sets/tsp-400/instance.txt"

insts = []
for source in [sourceFile1, sourceFile2]:
    if source is not None:
        with open(source, 'r') as f:
            insts.extend(f.read().strip().split())
if scenario == 'clasp-weighted_sequence':
    sample = []
    f = open(insFile, 'w+')
    for ins in insts:
        if "small" in ins:
            sample.append(ins)
    random.shuffle(sample)
    f.close()
else:
    sample = random.sample(insts, insNumer)
with open(insFile, 'w+') as f:
    for ins in sample:
        f.write(ins + '\n')
