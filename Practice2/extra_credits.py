'''
Tuning: 
    Tuning the hyperparameters and developing intuition for how they affect the final performance is a large part 
    of using Neural Networks, so we want you to get a lot of practice. Below, you should experiment with different 
    values of the various hyperparameters, including hidden layer size, learning rate, numer of training epochs, 
    and regularization strength. You might also consider tuning the learning rate decay, but you should be able to 
    get good performance using the default value.

Approximate results:
    You should be aim to achieve a classification accuracy of greater than 48% on the validation set. 
    Our best network gets over 52% on the validation set.

Experiment: 
    You goal in this exercise is to get as good of a result on CIFAR-10 as you can, with a fully-connected Neural Network. 
    For every 1% above 52% on the Test set we will award you with one extra bonus point. 
    Feel free implement your own techniques (e.g. PCA to reduce dimensionality, or adding dropout, or adding features to the solver, etc.).
'''
from __future__ import print_function
import os
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.data_utils import load_CIFAR10
from cs231n.vis_utils import visualize_grid
from utils import get_cifar10, report, run_tasks, makedirs
from data_loader import loader_data, preprocessing

def preprocess(X_train,X_val,X_test,X_dev):
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # As a sanity check, print out the shapes of the data
    print( 'Training data shape: ', X_train.shape)
    print( 'Validation data shape: ', X_val.shape)
    print( 'Test data shape: ', X_test.shape)
    print( 'dev data shape: ', X_dev.shape)

    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0) #3072 vector
    print( mean_image[:10]) # print a few of the elements

    # second: subtract the mean image from train, val, test, and dev data
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    print( X_train.shape, X_val.shape, X_test.shape, X_dev.shape)
    return X_train,X_val,X_test,X_dev

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.tight_layout()
    plt.savefig('./plots/extracredits_bestnetwork.png')

def example():
    X_train,y_train,X_val,y_val,X_dev,y_dev,X_test,y_test = loader_data()
    X_train,X_val,X_test,X_dev = preprocess(X_train,X_val,X_test,X_dev)
    input_size = 32 * 32 * 3
    hidden_size = 100
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
                num_iters=2000, batch_size=200,
                learning_rate=1e-4, learning_rate_decay=0.95,
                reg=0.5, verbose=True)

    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()
    print( 'Validation accuracy: ', val_acc)

    # Plot the loss function and train / validation accuracies
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.tight_layout()
    plt.savefig('./plots/extracredits_example.png')



################################################################################
#                       ACTUAL CODE STARTS FROM HERE                           #
################################################################################

X_train,y_train,X_val,y_val,X_dev,y_dev,X_test,y_test = loader_data()
X_train,X_val,X_test,X_dev = preprocess(X_train,X_val,X_test,X_dev)
best_net = None # store the best model into this 

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above in the example;                                            #
# these visualizations will have significant qualitative                        #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
results = {}
best_val = -1
best_net = None

#################################################################################
#                               BEGIN YOUR CODE                                 #
#                   Note: This section is for extra credit                      #    
#################################################################################
pass
################################################################################
#                              END OF YOUR CODE                                #
################################################################################
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print( 'lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print( 'best validation accuracy achieved during cross-validation: %f' % best_val) 
 
# visualize the weights of the best network
show_net_weights(best_net)
 
# Run on the test set
# When you are done experimenting, you should evaluate your 
# final trained network on the test set; you should get above 48%.

test_acc = (best_net.predict(X_test) == y_test).mean()
print( 'Test accuracy: ', test_acc)