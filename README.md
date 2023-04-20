# Parameter Optimization Of SVM

## Introduction
This assignment illustrates a way of finding the optimal hyperparameter of SVC. The data set used here is AI4I 2020 Predictive Maintenance Dataset Data Set from UCI repository.


## Dataset
The data used contains 14 columns and 10,000 rows. It is a synthetic dataset that reflects real predictive maintenance data encountered in industry. Out of 14 columns, the following 7 columns are used:
- Type
- Air temperature
- Process Tempature
- Rotational Speed
- Torque
- Tool wear
- Machine Failure
First 6 columns are used to predict the Machine Failure.

## Table for the SAMPLE DATASETS WITH ACCURACY AND SVM PARAMETERS
|Sample| Kernel   |   c | gamma   |   degree |   Accuracy |
|-----:|:---------|----:|:--------|---------:|-----------:|
|1     | rbf      |   7 | scale   |        1 |   0.976333 |
|2     | poly     |   2 | auto    |        5 |   0.978    |
|3     | rbf      |   4 | scale   |        4 |   0.981333 |
|4     | rbf      |   7 | scale   |        4 |   0.983333 |
|5     | rbf      |   6 | auto    |        3 |   0.981333 |
|6     | poly     |   3 | scale   |        5 |   0.980333 |
|7     | poly     |   6 | auto    |        5 |   0.982333 |
|8     | rbf      |   7 | scale   |        1 |   0.983667 |
|9     | rbf      |   7 | auto    |        3 |   0.980333 |
|10    | rbf      |   7 | auto    |        5 |   0.979333 |

## GRAPH BETWEEN ACCURACY AND ITERATIONS 
![100_iter](https://user-images.githubusercontent.com/79087337/233233178-e6059d45-f516-49ef-b6e7-cbf8374583d5.png)
