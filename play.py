import os
import sys
import gym
import cPickle as pickle
from os.path import exists, join
import numpy as np

def preprocess(img):
  img = img[35:195, ...]
  img = img[::2,::2,0]
  img[img == 144] = 0
  img[img == 109] = 0
  img[img != 0] = 1
  return img.astype(np.float32).ravel()

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))

def policy_forward(x):
  h = np.dot(model['W1'], x)
  h[h<0] = 0
  logp = np.dot(model['W2'], h)
  p = sigmoid(logp)
  return p, h

# model initialization
render = True
expt_dir = 'video/'
ckpt_path = join(expt_dir, 'ckpt')
expt_path = join(expt_dir, 'expt')

D = 80 * 80 # input dimensionality: 80x80 grid
max_games = 1
try:
  save = sys.argv[1]
except:
  raise AttributeError('model path expected: python play.py model.kl')

model = pickle.load(open(save, 'rb'))

if not exists(expt_dir):
  os.makedirs(expt_dir)
  expt_num = 1
else:
  expt_num = sorted(map(lambda k: int(k.split('-')[-1]),
                    [join(expt_dir, p) for p in os.listdir(expt_dir)
                     if os.path.isdir(join(expt_dir, p))]))[-1] + 1

env = gym.make("Pong-v0")
env.monitor.start('%s-%d' % (expt_path, expt_num))
observation = env.reset()
prev_x = None # used in computing the difference frame
episode_number = 0

try:
  while True:
    if render: env.render()

    cur_x = preprocess(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    pa, h = policy_forward(x)
    action = 2 if pa > 0.5 else 3
    y = 1 if action == 2 else 0

    observation, reward, done, _ = env.step(action)

    if done:
      episode_number += 1
      if episode_number == max_games:
        break
      env.reset()
except KeyboardInterrupt:
  pass

env.monitor.close()
