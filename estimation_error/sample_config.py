# sample configs from configuration space
# and save them at configs

from ConfigSpace.read_and_write import pcs

scenario = 'LKH-uniform-400'

if scenario == 'SATenstein-QCP':
    configNumber = 1000
    paramFile = "../target_algorithms/sat/satenstein/params.pcs"
    configFile = "./configs/SATenstein-QCP.txt"
elif scenario == 'CPLEX-REGIONS100':
    configNumber = 1000
    paramFile = "../target_algorithms/mip/cplex12.6/params.pcs"
    configFile = "./configs/CPLEX-REGIONS100.txt"
elif scenario == 'lpg-depots':
    configNumber = 1000
    paramFile = "../target_algorithms/planning/lpg/params.pcs"
    configFile = "./configs/lpg-depots.txt"
elif scenario == 'clasp-weighted_sequence':
    configNumber = 1000
    paramFile = "../target_algorithms/asp/clasp-3.1.4/params.pcs"
    configFile = "./configs/clasp-weighted_sequence.txt"
elif scenario == 'LKH-uniform-1000':
    configNumber = 1000
    paramFile = "../target_algorithms/tsp/LKH/params.pcs"
    configFile = "./configs/LKH-uniform-1000.txt"
elif scenario == 'LKH-uniform-400':
    configNumber = 1000
    paramFile = "../target_algorithms/tsp/LKH/params.pcs"
    configFile = "./configs/LKH-uniform-400.txt"

with open(paramFile, 'r') as f:
    configspace = pcs.read(f)
configFile = open(configFile, 'w+')

configs = []
configs.append(configspace.get_default_configuration())
configs.extend(configspace.sample_configuration(configNumber - 1))

for config in configs:
    paramDict = config.get_dictionary()
    cmdLine = ''
    for k, v in paramDict.items():
        cmdLine += "-%s %s " % (k, str(v))
    configFile.write(cmdLine + '\n')
configFile.close()
