import tensorflow as tf
import os
import sys
import urllib

HOME_FOLDER = "C:\Just Some Work\Practice\DeepLearning\Practice6\Homework6_Codebase\Part-A"

# create train folders
MNIST_FOLDER = os.path.join(HOME_FOLDER, "tf_mnist")
TRAIN_1_FOLDER = os.path.join(MNIST_FOLDER, "train-1")
TRAIN_2_FOLDER = os.path.join(MNIST_FOLDER, "train-2")
TRAIN_3_FOLDER = os.path.join(MNIST_FOLDER, "train-3")
if not os.path.isdir(MNIST_FOLDER):
    os.mkdir(MNIST_FOLDER)
if not os.path.isdir(TRAIN_1_FOLDER):
    os.mkdir(TRAIN_1_FOLDER)
if not os.path.isdir(TRAIN_2_FOLDER):
    os.mkdir(TRAIN_2_FOLDER)
if not os.path.isdir(TRAIN_3_FOLDER):
    os.mkdir(TRAIN_3_FOLDER)

# Resets the default graph
tf.compat.v1.reset_default_graph()

if sys.version_info[0] >= 3:
    from urllib3.request import urlretrieve
else:
    from urllib import urlretrieve

GITHUB_URL = 'https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'

### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=MNIST_FOLDER
                                                                 + '/data', one_hot=True)
### Get a sprite and labels file for the embedding projector ###
urlretrieve(GITHUB_URL + 'labels_1024.tsv', MNIST_FOLDER + '/labels_1024.tsv')
urlretrieve(GITHUB_URL + 'sprite_1024.png', MNIST_FOLDER + '/sprite_1024.png')


# Add convolution layer
def conv_layer(input, channels_in, channels_out):
    w = tf.Variable(tf.zeros([5, 5, channels_in, channels_out]))
    b = tf.Variable(tf.zeros([channels_out]))
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1],
                        padding="SAME")
    act = tf.nn.relu(conv + b)
    return act


# Add fully connected layer
def fc_layer(input, channels_in, channels_out):
    w = tf.Variable(tf.zeros([channels_in, channels_out]))
    b = tf.Variable(tf.zeros([channels_out]))
    act = tf.nn.relu(tf.matmul(input, w)) + b
    return act


# Setup placeholders, and reshape the data. Placeholders are used to feed data during the runtime.
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Create the network
conv1 = conv_layer(x_image, 1, 32)
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding="SAME")

conv2 = conv_layer(pool1, 32, 64)
pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1],
                       strides=[1, 2, 2, 1],
                       padding="SAME")
flattened = tf.reshape(pool2, [-1, 7 * 7 * 64])

fc1 = fc_layer(flattened, 7 * 7 * 64, 1024)
logits = fc_layer(fc1, 1024, 10)  # predictions

# Compute cross entropy as our loss function. Check the Tensorflow API for tf.reduce_mean and tf.nn.softmax_cross_entropy_with_logits_v2
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

# Use an AdamOptimizer to train the network
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

# compute the accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initialize all the variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# Train for 200 steps
for i in range(201):
    batch = mnist.train.next_batch(100)

    # Occasionally report accuracy
    if i % 10 == 0:
        [loss, train_accuracy] = sess.run([cost, accuracy],
                                          feed_dict={x: batch[0],
                                                     y: batch[1]})
        print("step %d, minibatch loss %g, training accuracy %g"
              % (i, loss, train_accuracy))

    # Run the training step
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

## Alternative way to train the network is like below. Uncomment the below and try to run. Understand the differences between two.
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     # Train for 200 steps
#     for i in range(201):
#         batch = mnist.train.next_batch(100)

#         # Occasionally report accuracy
#         if i % 10 == 0:
#             [loss, train_accuracy] = sess.run([cost, accuracy],
#                                         feed_dict={x: batch[0],
#                                                     y: batch[1]})
#             print("step %d, minibatch loss %g, training accuracy %g"
#                     % (i, loss, train_accuracy))

#         # Run the training step
#         sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})
