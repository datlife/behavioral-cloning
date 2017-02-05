import tensorflow as tf
from constants import *
slim = tf.contrib.slim


def apply_vision_simple(image, keep_prob, batch_size, seq_len, scope=None, reuse=None):
    layer_norm = lambda x: tf.contrib.layers.layer_norm(inputs=x, center=True, scale=True,
                                                        activation_fn=None, trainable=True)

    video = tf.reshape(image, shape=[batch_size, LEFT_CONTEXT + seq_len, HEIGHT, WIDTH, CHANNELS])

    with tf.variable_scope(scope, 'Vision', [image], reuse=reuse):

        net = slim.convolution(video, num_outputs=64, kernel_size=[3, 12, 12], stride=[1, 6, 6], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux1 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
                                    activation_fn=None)

        net = slim.convolution(net, num_outputs=64, kernel_size=[2, 5, 5], stride=[1, 2, 2], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux2 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
                                    activation_fn=None)

        net = slim.convolution(net, num_outputs=64, kernel_size=[2, 5, 5], stride=[1, 1, 1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)
        aux3 = slim.fully_connected(tf.reshape(net[:, -seq_len:, :, :, :], [batch_size, seq_len, -1]), 128,
                                    activation_fn=None)

        net = slim.convolution(net, num_outputs=64, kernel_size=[2, 5, 5], stride=[1, 1, 1], padding="VALID")
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)

        # at this point the tensor 'net' is of shape batch_size x seq_len x ...
        aux4 = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 128, activation_fn=None)

        net = slim.fully_connected(tf.reshape(net, [batch_size, seq_len, -1]), 1024, activation_fn=tf.nn.relu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)

        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)

        net = slim.fully_connected(net, 256, activation_fn=tf.nn.relu)
        net = tf.nn.dropout(x=net, keep_prob=keep_prob)

        net = slim.fully_connected(net, 128, activation_fn=None)
        # aux[1-4] are residual connections (shortcuts)
        return layer_norm(tf.nn.elu(net + aux1 + aux2 + aux3 + aux4))
