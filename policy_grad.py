import tensorflow as tf
import numpy as np
import gym
import os
from tqdm import *

from model import nn_model, conv_model

# threads = 2  # num of environs to run in parallel
render = False
learning_rate = 0.0001
batch_size = 10
width = height = 160
gamma = 0.99
num_episode = 0
experience = 100000

def discount_rewards(r):
    disc = np.zeros_like(r)
    acc = 0
    for i in reversed(xrange(disc.size)):
        if r[i]:
            acc = 0
        acc = gamma * acc + r[i]
        disc[i] = acc
    disc -= np.mean(disc)
    disc /= (np.std(disc) + 1e-6)
    return disc

def preprocess(img):
  img = img[35:195, ...]
  img = img[:,:,0]
  img[img == 144] = 0
  img[img == 109] = 0
  img[img != 0] = 1
  return img.astype(np.float32)


# set up placeholders for observations, actions and rewards
obsv_op = tf.placeholder(tf.float32, shape=[None, width, height, 1],
                         name='observation')
target_op = tf.placeholder(tf.float32, shape=[None], name='action')
reward_op = tf.placeholder(tf.float32, shape=[None], name='rewards')

# create NN model
train_op, loss, grads_and_vars, y = nn_model(obsv_op, target_op, reward_op)
grads = map(lambda k: k[0], grads_and_vars)
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

env = gym.make('Pong-v0')
obs = env.reset()

prev = np.zeros_like(preprocess(obs), np.float32)
epreward = 0.
decisions = []
actions = []
rewards = []
disc_rewards = []
xs = []

while True:
    if render: env.render()
    curr = preprocess(obs)
    diff = curr - prev
    diff[curr == 1.] = 1.
    diff = diff[:, :, np.newaxis]

    pa = sess.run(y, feed_dict={obsv_op: [diff]})
    a = 2 if pa > np.random.uniform() else 3
    t = 1. if a == 2 else 0.

    xs.append(diff)
    decisions.append(pa.ravel())
    actions.append(t)

    obs, r, epdone, _ = env.step(a)
    epreward += r

    rewards.append(r)

    if epdone:
        num_episode += 1

        disc_rewards.extend(discount_rewards(rewards))

        if num_episode % batch_size == 0:
            # train
            num_samples = len(xs)
            print "Training model... Fitting %d samples" % num_samples
            for start, end in tqdm(zip(xrange(0, num_samples, 32),
                                       xrange(32, num_samples, 32))):
                batch_xs = np.asarray(xs[start:end])
                batch_rs = np.asarray(disc_rewards[start:end])
                batch_ts = np.asarray(actions[start:end])
                batch_ys = np.asarray(decisions[start:end])
                feed_dict={
                    obsv_op: batch_xs, y: batch_ys,
                    target_op: batch_ts, reward_op: batch_rs
                }
                sess.run(train_op, feed_dict=feed_dict)
                # gs = sess.run(grads, feed_dict=feed_dict)
            xs, decisions, actions, disc_rewards = [], [], [], []

        rewards = []
        print "Match complete. Resetting env."
        obs = env.reset()
    if r:
        print "Episode: %d Game Reward: %s" % (num_episode,
                                               " 1\t##" if r == 1 else "-1")
    if num_episode > experience:
        break
