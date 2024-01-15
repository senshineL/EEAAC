# On Performance Estimation in Automatic Algorithm Configuration (ac_estimation_error)

The whole project is developed based on [Algorithm Configuration Library 2.0](https://bitbucket.org/mlindauer/aclib2/)

## Install
ac_estimation_error requires Python 3.5 (we implemented under Anaconda 4.5.11)

pip install -r requirements.txt

Some target algorithms may have further dependencies.

## Installation of Instances
Since the instance sets are by far too large to upload,  please download the instance sets manually.
Please extract the instances in the root directory of ac_estimation_error:
`tar xvfz XXX.tar.gz`

* [sat_QCP](http://aad.informatik.uni-freiburg.de/~lindauer/aclib/sat_QCP.tar.gz)
* [asp_weighted-sequence](http://aad.informatik.uni-freiburg.de/~lindauer/aclib/asp_weighted-sequence.tar.gz)
* [LKH-uniform-400](https://drive.google.com/open?id=1uZwH13xENnaYu-ul_BplW4cm4hPNzU4k)
* [LKH-uniform-1000](https://drive.google.com/open?id=1IIDmwrMfByiVTOv1ox54Rt_A6iVDqRbO)

## Gather Performance
To gather performance for each considered scenario
```
cd estimation_error
python gather_PM.py scenario
```
Or you can query the usage of gather_PM.py by
```
cd estimation_error
python -h gather_PM.py
```
The performance matrix (named PM.npy) is stored in estimation_error/archive/scenario/

## Comparison of Different Estimators
To compare different estimators
```
cd estimation_error
python compare_estimators.py
```
Or you can query the usage of compare_estimators.py by
```
cd estimation_error
python -h compare_estimators.py
```
The result (named VM.npy) is stored in estimation_error/archive/scenario/
## Estimation Error at different m, N, K
To obtain estimation error on different m, N and K
```
cd estimation_error
python deviation_m.py scenario
python deviation_N.py scenario
python deviation_K.py scenario
```
The result (named DM_m.npy, DM_N.npy and DM_K.npy) is stored in estimation_error/archive/scenario/

## Notes for Computational Budgets
Since the above experiments could be time-consuming, we suggest using parallelism by setting --max_parallism when calling these scripts.

