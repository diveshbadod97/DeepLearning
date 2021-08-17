import tensorflow as tf

# Use tf.while_loop and tf.cond to implement question a and b in this file respectively.

# a) Write the following while loop in TensorFlow:
'''
i = 10
while (i > 0):
    i--
print(i)
'''
# Your TF code below. Show code and output in latex file.
i = tf.constant(10)
c = lambda i: i > 0
b = lambda i: (tf.subtract(i, 1),)
r = tf.while_loop(c, b, [i])
print('The answer to part a is :')
print(r)
# b) Write the following conditional state statement in TensorFlow:
'''
x = 5.0
y = 2.0
if (x > y):
    print (x - 2.0)
else:
    print (y / 2.0)

'''
x = tf.constant(5.0)
y = tf.constant(2.0)


def f1():
    return tf.subtract(x, 2.0)


def f2():
    return tf.divide(y, 2.0)


r = tf.cond(tf.greater(x, y), f1, f2)
# Your TF code below. Show code and output in latex file.
print('The answer to part b is :')
print(r)
