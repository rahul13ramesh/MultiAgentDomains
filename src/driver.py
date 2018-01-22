#!/usr/bin/env python3

from simple_reference.env import GridEnv as env1
from simple_cover.env import GridEnv as env2
from simple_chaser.env import GridEnv as env3
from simple_speaker_listener.env import GridEnv as env4
import numpy as np

envs1 = env1(gridSize=20, nAgents=4, sparseReward=False)
envs2 = env2(gridSize=20, nAgents=4)
envs3 = env3(gridSize=20, nAgents=4)
envs4 = env4(gridSize=20, landMarks=3)

#  for i in range(50):
    #  envs1.act(np.random.randint(0, 5, 4), viz=True)

#  for i in range(50):
    #  envs2.act(np.random.randint(0, 5, 4), viz=True)

#  for i in range(50):
    #  envs3.act(np.random.randint(0, 5, 4), viz=True)

for i in range(50):
    envs4.act(np.random.randint(0,5), viz=True)
