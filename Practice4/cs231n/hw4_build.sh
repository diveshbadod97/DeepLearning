#!/bin/bash
#

# This is a sample bash script which you will be using
# to submit jobs to the cluster for running. DO NOT CHANGE anything other than where
# specified if you don't know what you're doing.

# Change the job name from test to something unique everytime you run this script to
# submit a job
#SBATCH -J hw3_build

# Standard output and error files. Ideally should be same as job name.
#SBATCH -o hw3_build.out 
#SBATCH -e hw3_build.err

# Instructions to SLURM to run your job on the correct partition without
# hogging too much resources. DO NOT CHANGE THESE LINES
#SBATCH -t 5
#SBATCH -p kgcoe-gpu -n 2 --mem=2048 --qos=free



module load python/3.5.2

# The following line runs your job script. Make sure this script is in the same directory
# as your python scripts. The arguments at the end refer to the task number you want to run.
# Use the argument number corresponding to the task you want to run. Use 'all' if you want
# to run all the tasks at once. For example to run all tasks in Homework 1 Part A type:
#python3 hw1a.py --run all
python3 setup.py build_ext --inplace
