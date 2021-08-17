import numpy as np
import pdb
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

'''

We shall implement linear regression in this file

'''
# Creating the dataset. (No coding required)
x = np.linspace(-1, 1, 100)
y = x * 3 + np.random.randn(x.shape[0]) * 0.7

# Visualizing the dataset. (No coding required)
plt.plot(x, y, 'ro')
plt.savefig('./scatterplot.png')

# Step 2 : Create 2 placeholders X(for the input) and Y(for the output) using tf.placeholder. The dtype should be tf.float32 and the name should be 'x' and 'y' respectively. (Coding required)
X = tf.compat.v1.placeholder(tf.float32, name='x')
Y = tf.compat.v1.placeholder(tf.float32, name='y')

# Step 3: Initialize 2 variables 'w'(for the weights) and 'b'(for the bias) using tf.Variable. Their values are 0.0 and names should be 'weights' and 'bias' respectively. (Coding required)
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')
# Step 4: Construct a model to predict Y. The model is: predicted_Y = X*w + b.  (Coding required)
predicted_Y = X * w + b

# Step 5: Calculate the square error using tf.square and tf.subtract and store it in the variable loss. Assign the name 'loss' (Coding required)
# The loss is given by the equation :: loss = (Y-predicted_Y)^2
loss = tf.square(tf.subtract(Y, predicted_Y))

# Step 6: Define the optimizer. We use the tf.train.GradientDescentOptimizer function (No coding required)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# Step 7: Run the session.  (Coding required)
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True

num_epochs = 10

with tf.Session(config=session_config) as sess:
    # initialize the variables that are going to be learned. We declared w and b as the variables and hence they will get updated during this sess run.
    sess.run(tf.global_variables_initializer())

    # Train the model for num_epochs. Inside the sess.run([]) pass in the GradientDescentOptimizer we created.
    for i in range(num_epochs):
        sess.run([optimizer], feed_dict={X: x, Y: y})

        # Retrieve the updated values of w and b.
        curr_w, curr_b = sess.run([w, b], {X: x, Y: y})
        print("Epoch: {} | Current W: {} | Current Bias: {} | ".format(i, curr_w, curr_b))

# Step 8: Plot your final model.
# Plot a line over the data points using the last value of w and b you get on your screen(at epoch 24).
final_w = curr_w
final_bias = curr_b
predicted_y = x * final_w + final_bias
plt.plot(x, predicted_y, label="Final Line")
plt.title(final_w)
plt.legend()
plt.savefig('final_line.png')

# Step 9: Plot your initial model.
# Plot a line over the data points using the first value of w and b you get on your screen(at epoch 0).
init_w = 0.0
init_bias = 0.0
init_y = x * init_w + init_bias
plt.plot(x, init_y, label='Initial Line ')
plt.title(init_w)
plt.legend()
plt.savefig('init_line.png')