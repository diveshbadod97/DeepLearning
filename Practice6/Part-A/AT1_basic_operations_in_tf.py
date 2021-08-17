# These are some of the basic operations in tensorflow.
import tensorflow as tf
import numpy as np

# import tensorflow as tf                         # import tensorflow 
# node1 = tf.constant([[3.0, 2.0]], tf.float32)   # define a constant
# with tf.Session() as sess                       # declare a session
# sess.run([tf.global_variables_initializer()])   # initialize the global variable initializer
# print (node1.eval())                            # evaluate the tensor and print its value
# print (node1.get_shape())                       # print shape
# print (node1.dtype)                             # print type


#################################################################################
# a. Store the value "tf.constant([3.0])" in a variable. Print its shape, data type and value in that order.
print('The answer to Part a are as below: ')
### YOUR ANSWER HERE. Show this code  and output of this task on the latex file.

a = tf.constant([[3.0]], dtype=tf.float32)
print(a)
# b. Store the value "tf.constant([3])" in a variable. Print its shape, data type and value in that order.
print('The answer to Part b are as below: ')
### YOUR ANSWER HERE. Show this code  and output of this task on the latex file.
b = tf.constant([3])
print(b)
# c. Create a 3 x 4 matrix with all its elements = 1. Store in a variable. Print its shape, data type and value in that order.
print('The answer to Part c are as below: ')

### YOUR ANSWER HERE. Show this code  and output of this task on the latex file.

c = tf.ones([3, 4])
print(c)
# d. Create the below depicted 3x5 array using numpy.
'''
[1 2 3 5 7
11 13 17 19 23
29 31 27 41 43]
'''
# Print the numpy array. Show this code on the latex file
# Convert this array to a tensor of constant value. Show this code on the latex file
# Print its shape, data type and value in that order

print('The answer to Part d are as below: ')
### YOUR ANSWER HERE. Show this code  and output of this task on the latex file.
d = np.array([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 27, 41, 43]).reshape(3, 5)
print(d)
# e. Define a 3x3 matrix of int type consisting of the first 9 prime numbers. Fill this matrix in the column major order. https://ipfs.io/ipfs/QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco/wiki/Row-major_order.html
# Define another 3x3 matrix of int type consisting the first 9 fibbonacci numbers. Fill this matrix in the row major order.
# Perform element wise multiplication using tensorflow.
# Show both input matrices and output matrices in latex.
print('The answer to Part e are as below: ')
e1 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23]).reshape((3, 3), order='F')
e2 = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21]).reshape((3, 3), order="C")
e1 = tf.convert_to_tensor(e1)
e2 = tf.convert_to_tensor(e2)
print(e1)
print(e2)
print(tf.matmul(e1, e2))
# f. Define a 3x3 matrix consisting of the first 9 prime numbers. Fill this matrix in the column major order.
# Define another 3x3 matrix consisting the first 9 fibbonacci numbers. Fill this matrix in the row major order.
# Perform the matrix multiplication : prime matrix * fibo matrix
# Show both the matrices, the multiplication step and output in latex.
print('The answer to Part f are as below: ')
f1 = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23]).reshape((3, 3), order='F')
f2 = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21]).reshape((3, 3), order="C")
f3 = f1 * f2
print(f3)
