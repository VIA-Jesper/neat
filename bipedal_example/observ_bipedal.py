## test

import os
import pickle
import neat
import gym
import numpy as np

with open('winner_bipedal', 'rb') as f:
    c = pickle.load(f)

print('loaded genome:')
print(c)

## load the config file
config_path = os.path.join('config_bipedal', )
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)
net = neat.nn.FeedForwardNetwork.create(c, config)

env = gym.make('BipedalWalker-v3')
observation = env.reset()

done = False
while not done:
    action = net.activate(observation)

    observation, reward, done, info = env.step(action)
    env.render()

env.close()