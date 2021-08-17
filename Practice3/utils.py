import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import subprocess

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

def get_cifar10():
    # get cifar-10 dataset
    os.chdir("./cs231n/datasets/")
    os.chmod("get_datasets.sh", 436)
    os.system("sh ./get_datasets.sh")
    os.chdir("../../")

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run_tasks(run_tasks):
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='all', help='Run the tasks')
    args = parser.parse_args()
    run_str = args.run
    run_list =[]
    if run_str=='all':
        run_list = [1 for val in run_tasks]
    else:
        run_list = [0 for val in run_tasks]
        for i in run_str:
            if int(i)<=len(run_tasks):
                str_task = 'task'+str(i)
                idx = run_tasks.index(str_task)
                run_list[idx]=1
    return run_list