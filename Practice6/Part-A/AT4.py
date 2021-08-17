import tensorflow as tf

tf.enable_eager_execution()
import numpy as np


def dogslikescats(num_max):

    while count_initial < num_max:
        if count_initial % 3 == 0 and count_initial % 5 == 0:
            print('dogslikescats')
        if count_initial % 3 == 0:
            print('dogs')
        if count_initial % 5 == 0:
            print('cats')


if __name__ == '__main__':
    dogslikescats(20)