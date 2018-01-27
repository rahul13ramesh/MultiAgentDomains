#!/usr/bin/env python3

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gridEnvironment import MultiAgentGrid


class GridEnv(MultiAgentGrid):

    def __init__(self, gridSize=7, nAgents=2, sparseReward=True, seed=None):
        self.gridSize = gridSize
        self.numAgents = nAgents
        self.sparse = sparseReward
        self.defineMoves()
        self.reset_world(nAgents, gridSize)

        if seed is not None:
            np.random.seed(seed)

    def reset_world(self, numAgents, gridsize):
        self.agents = []
        self.landmarks = []
        self.steps = 0

        for i in range(numAgents):
            self.agents.append(np.random.randint(low=0,
                                                 high=gridsize,
                                                 size=2))
        for i in range(numAgents):
            self.landmarks.append(np.random.randint(low=0,
                                                    high=gridsize,
                                                    size=2))

        self.agentReached = [False for i in range(numAgents)]

    def getState(self, agentNum, addId=True):
        """
        State is the relative positions of all other Landmarks and agents
        """

        relPositions = []
        for i in range(self.numAgents):
            if(i != agentNum):
                relPositions.append(self.agents[i] - self.agents[agentNum])
                relPositions.append(self.landmarks[i] - self.agents[agentNum])

        relPositions.append(self.agents[agentNum])

        if addId:
            relPositions.append([agentNum])

        return np.concatenate(relPositions)

    def act(self, actions, viz=False):
        """
        Returns Reward
        Reward of +1 first time designated landmark visited for agent
        Reward of all agets summed together
        """

        if len(actions) != self.numAgents:
            raise ValueError("Action size is incorrect")

        self.steps += 1
        reward = 0.0

        #  Update states of all agents
        for ind, act in enumerate(actions):
            if act == self.NOOP:
                pass
            elif act == self.LEFT:
                self.agents[ind] = (self.agents[ind] + [0, -1] +
                                    self.gridSize) % self.gridSize
            elif act == self.RIGHT:
                self.agents[ind] = (self.agents[ind] + [0, 1] +
                                    self.gridSize) % self.gridSize
            elif act == self.DOWN:
                self.agents[ind] = (self.agents[ind] + [1, 0] +
                                    self.gridSize) % self.gridSize
            elif act == self.UP:
                self.agents[ind] = (self.agents[ind] + [-1, 0] +
                                    self.gridSize) % self.gridSize

        if self.sparse:
            for i in range(self.numAgents):
                #  Reach landmark
                if not self.agentReached[i]:
                    if np.sum(self.agents[i] == self.landmarks[i]) == 2:
                        reward = reward + 1.0
                        self.agentReached[i] = True
                #  Once landmark Reached, stay there
                if self.agentReached[i]:
                    if np.sum(self.agents[i] == self.landmarks[i]) == 2:
                        reward = reward + 0.05
        else:
            for i in range(self.numAgents):
                x1, y1 = self.agents[i]
                x2, y2 = self.landmarks[i]

                xdist = min(abs(x1 - x2), self.gridSize - abs(x1 - x2))
                ydist = min(abs(y1 - y2), self.gridSize - abs(y1 - y2))
                reward = reward - xdist - ydist

            reward = reward / self.gridSize

        if viz:
            self.visualizeState()

        return reward

    def visualizeState(self):
        if self.numAgents >= 7:
            raise ValueError("Visualization works for only if No." +
                             "of Agents < 7")

        N = self.gridSize
        # make an empty data set
        data = np.ones((N, N)) * np.nan

        for i in range(self.numAgents):
            x1, y1 = self.agents[i]
            x2, y2 = self.landmarks[i]
            data[x1][y1] = 2 * i
            data[x2][y2] = 2 * i + 1

        colors = [
            '#EC7063',
            '#E74C3C',
            '#AF7AC5',
            '#9B59B6',
            '#5499C7',
            '#2980B9',
            '#1ABC9C',
            '#17A589',
            '#F4D03F',
            '#F1C40F',
            '#808B96',
            '#566573']

        colors = colors[0: 2 * self.numAgents]

        # make a figure + axes
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        # make color map
        my_cmap = matplotlib.colors.ListedColormap(colors)
        # set the 'bad' values (nan) to be white and transparent
        my_cmap.set_bad(color='w', alpha=0)
        # draw the grid
        for x in range(N + 1):
            ax.axhline(x, lw=2, color='k', zorder=5)
            ax.axvline(x, lw=2, color='k', zorder=5)
        # draw the boxes
        ax.imshow(data, interpolation='none', cmap=my_cmap,
                  extent=[0, N, 0, N], zorder=0)
        # turn off the axis labels
        ax.axis('off')
        plt.savefig("img_ref/" + str(self.steps) + ".png")
        plt.close()
