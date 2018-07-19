import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#from tensorflow.examples.tutorial.mnist import input_data
import input_data
mnist = input_data.read_data_sets("./mnist/data", one_hot = True) #Use MNIST Dataset
X = tf.placeholder(tf.float32, [None, 28 * 28]) # MNIST Image (28*28)
Z = tf.placeholder(tf.float32, [None, 128]) # Noise Z (Dimension = 128)

epoch_val = 200
batch_size = 100

# ============================= Generator =============================

GEN_W1 = tf.Variable(tf.random_normal([128, 256], stddev = 0.01))
GEN_W2 = tf.Variable(tf.random_normal([256, 28*28], stddev = 0.01))
GEN_b1 = tf.Variable(tf.zeros([256]))
GEN_b2 = tf.Variable(tf.zeros([28*28]))

def generator(noise_input):
    hidden = tf.nn.relu(tf.matmul(noise_input, GEN_W1) + GEN_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, GEN_W2) + GEN_b2)
    return output

# ============================= Discriminator =============================

DIS_W1 = tf.Variable(tf.random_normal([28*28, 256], stddev = 0.01))
DIS_W2 = tf.Variable(tf.random_normal([256, 1], stddev = 0.01))
DIS_b1 = tf.Variable(tf.zeros([256]))
DIS_b2 = tf.Variable(tf.zeros([1]))

def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, DIS_W1) + DIS_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, DIS_W2) + DIS_b2)
    return output

#loss
G = generator(Z)

loss_GEN = -tf.reduce_mean(tf.log(discriminator(G)))
loss_DIS = -tf.reduce_mean(tf.log(discriminator(X)) + tf.log(1 - discriminator(G)))

train_DIS = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_DIS, var_list=[DIS_W1, DIS_b1, DIS_W2, DIS_b2])
train_GEN = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss_GEN, var_list=[GEN_W1, GEN_b1, GEN_W2, GEN_b2])

#session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# ============================= Train, Test =============================

noise_test = np.random.normal(size = (10, 128)) #Test Sample, Noise Dimension
for epoch in range(epoch_val):
    print("Epoch: (%3d / %3d)" % (epoch, epoch_val))
    for i in range(int(mnist.train.num_examples / batch_size)):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        noise = np.random.normal(size=(100,128))

        sess.run(train_DIS, feed_dict={X: batch_xs, Z: noise})
        sess.run(train_GEN, feed_dict={Z: noise})

    if epoch == 0 or (epoch + 1) % 10 == 0:
        samples = sess.run(G, feed_dict={Z: noise_test})
        fig, ax = plt.subplots(1, 10, figsize=(10, 1))
        for i in range(10):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))
        plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)
