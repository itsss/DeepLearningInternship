from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)
num_steps = 20000
batch_size = 32

image_dim = 784 
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 


def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tf.layers.dense(x, units=6 * 6 * 128)
        x = tf.nn.tanh(x)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        x = tf.nn.sigmoid(x)
        return x

def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        x = tf.layers.dense(x, 2)
    return x

noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
gen_sample = generator(noise_input)

dis_real = discriminator(real_image_input)
dis_fake = discriminator(gen_sample, reuse=True)
dis_concat = tf.concat([dis_real, dis_fake], axis = 0)

stacked_gan = discriminator(gen_sample, reuse=True)

dis_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

dis_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = dis_concat, labels = dis_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = stacked_gan, labels = gen_target))

optimizer_gen = tf.train.AdamOptimizer(learning_rate = 0.001)
optimizer_dis = tf.train.AdamOptimizer(learning_rate = 0.001)

gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_dis = optimizer_dis.minimize(dis_loss, var_list=dis_vars)

init = tf.global_variables_initializer()
# ====================== TRAINING ======================
with tf.Session() as sess:
    sess.run(init)
    for i in range(1, num_steps+1):
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])

        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        batchd_y = np.concatenate([np.ones([batch_size]), np.zeros([batch_size])], axis=0)
        batchg_y = np.ones([batch_size])

        feed_dict = {real_image_input: batch_x, noise_input: z,
                     dis_target: batchd_y, gen_target: batchg_y}
        _, _, g_loss, d_loss = sess.run([train_gen, train_dis, gen_loss, dis_loss],
feed_dict=feed_dict)
        if i % 100 == 0 or i == 1:
            print('(%i/%i) Generator Loss: %f / Discriminator Loss: %f' % (i, num_steps, g_loss, d_loss))

    f, a = plt.subplots(4, 10, figsize=(10, 4))
    for i in range(10):
        z = np.random.uniform(-1., 1., size=[4, noise_dim])
        g = sess.run(gen_sample, feed_dict={noise_input: z})
        for j in range(4):
            img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3))
            a[j][i].imshow(img)
    print("Training Finished!")

    f.show()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('result_'+str(num_steps)+' steps.pdf')
    plt.waitforbuttonpress()
