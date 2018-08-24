# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-08-24 10:00:02
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-08-24 14:42:41
#
# -----------------------------------------
#
# Gridworld: NxN gridworld, with 5 actions per state corresponding
# to moving in each direction and staying in place.
# In deterministic mode: actions succed
# In non-deterministic: there is a probability that actions fail
# Reward are 0 all the time, except when the agent reach the final state, the reward is +10

import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy

import utils

class Gridworld(object):
    """GridWorld envirnoment for RL and IRL algorithms testing and evaluation"""

    def __init__(self, size, prob_trans=1):

        # Create the grid of reward: 0 everywhere except at the ending state (bottom right)
        self.grid_size = size;
        self.grid = np.zeros([size,size]);
        self.grid[size-1, size-1] = 10;

        # (null, north, east, south, west)
        self.actions = ((0,0), (-1,0), (0,1), (1,0), (0,-1));

        # Transition probability
        self.prob_trans = prob_trans;

        # Terminal state: bottom right corner of the grid
        self.terminal_state = (size-1, size-1);

        # Current state of the agent
        self._current_state = (0,0);


        # State log
        self._state_log = [];

    """
    Return the reward corresponding to the specified state
    """
    def _get_reward(self, state):
        return self.grid[state[0], state[1]];

    """
    Return true of the agents reached the terminal state
    """
    def is_over(self, state):
        return state == self.terminal_state;

    """
    Set the start state of the agent
    """
    def set_start_state(self, start_state):
        self._current_state = start_state;

    """
    Move the agent to the new state, keep track of states
    """
    def move_current_state(self, new_state):
        self._state_log.append(new_state);
        self._current_state = new_state;


    """
    Return the state log, which coresponds to the states visited by the agents
    """
    def get_state_log(self):
        return self._state_log


    """
    Check if the state is within the grid
    Return true of false
    """
    def is_in_grid(self, state):

        flag = state[0] >= 0 and state[0] < self.grid_size and state[1] >= 0 and state[1] < self.grid_size;

        return flag

    """
    Compute new state and reward according to the agent's action, return the new state
    In case of a non-determnistic MDP, the actions as a probability of failing of prob_trans
    If the action fails, an other action is execute with a random probability

    Returns:
        is_done
        new_state
        reward
    """
    def take_action(self, action_id):

        # Do not move is at terminal state
        if self.is_over(self._current_state):
            return True, self._current_state, 0;

        # Generate the probability of chosing an action according to:
        # - action_id: action asked
        # - prob_trans: transition probability
        prob_action = np.ones([5])*(float(1-self.prob_trans)/4);
        prob_action[action_id] = self.prob_trans;

        # Sample action
        action_exec_id = np.random.choice(np.arange(0,5), 1, True, prob_action);
        action_exec_id = int(action_exec_id);

        # Move the agent according to the executed action
        action_exec = self.actions[action_exec_id];
        new_state = (self._current_state[0] + action_exec[0], self._current_state[1] + action_exec[1]);

        # Ensure new state remains in the grid, do not move the agent if outside the grid
        if self.is_in_grid(new_state):
            self.move_current_state(new_state);
            reward = self._get_reward(new_state)
            print(reward)

        # If outside, do not move agent and return reward of -1
        else:
            reward = -1;

        return False, self._current_state, reward;


if __name__ == "__main__":

    # Create gridworld
    gw = Gridworld(10, 0.5);

    # Take some actions
    for i in range(10):
        done_flag, st, rw = gw.take_action(2);
        done_flag, st, rw = gw.take_action(3);

    grid_path = copy.copy(gw.grid);
    for state_visited in gw.get_state_log():
        grid_path[state_visited[0],state_visited[1]] = state_visited[0]+state_visited[1];

    # plots
    plt.figure()
    plt.subplot(1, 2, 1)
    utils.plot_heatmap(gw.grid, "Reward Map", False)
    plt.subplot(1, 2, 2)
    utils.plot_heatmap(grid_path, "Agent path", False)

    plt.show()









