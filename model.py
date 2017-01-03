import tensorflow as tf
import numpy as np

learning_rate = 1e-2
clip_val = 10.

def conv2d(x, k, n, strides=1, scope='conv'):
    with tf.variable_scope(scope) as sc:
        c = int(x.get_shape()[-1])
        W = get_weights([k, k, c, n])
        b = get_biases(n)
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME', name='preact')
        x = tf.nn.bias_add(x, b)
        o = tf.nn.relu(x, name='activation')
    return o

def fc_layer(x, h, activation='', dropout=False, scope='fc'):
    with tf.variable_scope(scope) as sc:
        assert len(x.get_shape().as_list()) == 2
        c = int(x.get_shape()[-1])
        W = get_weights([c, h])
        o = tf.matmul(x, W, name='preact')
        if activation == 'relu':
            o = tf.nn.relu(o, name='activation')
        elif activation == 'sigmoid':
            o = tf.sigmoid(o, name='activation')
        if dropout:
            o = tf.nn.dropout(o, tf.constant(0.5), name='dropout')
    return o

def flatten(x):
    dim = np.prod(x.get_shape().as_list()[1:])
    return tf.reshape(x, [-1, dim])

def get_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1), name='weights')

def get_biases(n):
    return tf.Variable(tf.zeros([n]), name='biases')

def conv_model(obsv, target, rewards):
    c1 = conv2d(obsv, 3, 32, strides=1, scope='conv1')
    c2 = conv2d(c1, 3, 64, strides=1, scope='conv2')
    c3 = conv2d(c2, 3, 24, strides=2, scope='conv3')
    f = flatten(c3)
    fc1 = fc_layer(f, 100, activation='relu', scope='fc1')
    y = fc_layer(fc1, 1, activation='sigmoid', scope='fc2')

    loss = tf.constant(0.5) * tf.reduce_sum(tf.mul(rewards, (target - y)**2))
    optim = tf.train.GradientDescentOptimizer(learning_rate, name='optim')
    grads_and_vars = optim.compute_gradients(loss)
    grads_and_vars = [(tf.clip_by_value(g, -clip_val, clip_val), v)
                      for (g, v) in grads_and_vars]
    train_op = optim.apply_gradients(grads_and_vars)

    return train_op, loss, grads_and_vars, y


def nn_model(obsv, target, rewards):
    x = flatten(obsv)
    fc1 = fc_layer(x, 200, activation='relu', scope='fc1')
    y = fc_layer(fc1, 1, activation='sigmoid', scope='fc2')
    assert y.get_shape().as_list()[0] == target.get_shape().as_list()[0]

    loss = tf.constant(0.5) * tf.reduce_sum(tf.mul(rewards, (y - target)**2))
    optim = tf.train.GradientDescentOptimizer(learning_rate, name='optim')
    grads_and_vars = optim.compute_gradients(loss)
    grads_and_vars = [(tf.clip_by_value(g, -clip_val, clip_val), v)
                      for (g, v) in grads_and_vars]
    train_op = optim.apply_gradients(grads_and_vars)

    return train_op, loss, grads_and_vars, y
