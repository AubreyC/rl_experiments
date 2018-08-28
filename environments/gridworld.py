# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-08-24 10:00:02
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-08-28 15:58:52
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

        print("Gridworld:\n");
        print("size: {}x{}, trans prob: {}".format(size, size, prob_trans))

        # Create the grid of reward: 0 everywhere except at the ending state (bottom right)
        self.grid_size = size;
        self.grid = np.zeros([size,size]);

        # Define terminal state:
        self.terminal_state = (size-1, size-1);

        # Set high reward at terminal state
        self.grid[size-1, size-1] = 10;

        # N total states
        self.n_states= self.grid.shape[0]*self.grid.shape[1];

        # (null, north, east, south, west)
        self.actions = ((0,0), (-1,0), (0,1), (1,0), (0,-1));
        self.n_actions = len(self.actions);

        # Transition probability
        self.prob_trans = prob_trans;

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
    Return the curretn state
    """
    def get_current_state(self):
        return self._current_state;


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
        prob_action = np.ones([self.n_actions])*(float(1-self.prob_trans)/self.n_actions);
        prob_action[action_id] = prob_action[action_id] + self.prob_trans;

        # Sample action
        action_exec_id = np.random.choice(np.arange(0,self.n_actions), 1, True, prob_action);
        action_exec_id = int(action_exec_id);

        # Move the agent according to the executed action
        action_exec = self.actions[action_exec_id];
        new_state = (self._current_state[0] + action_exec[0], self._current_state[1] + action_exec[1]);

        # Ensure new state remains in the grid, do not move the agent if outside the grid
        if self.is_in_grid(new_state):
            self.move_current_state(new_state);
        #     reward = self._get_reward(self._current_state)
        #     print(reward)

        # # If outside, do not move agent and return reward of -1
        # else:
        #     reward = -1;

        reward = self._get_reward(self._current_state)
        # print('Reward: {}'.format(reward))

        return False, self._current_state, reward;

    """
    Compute the probability of reaching s2_state from s1_state if action_id is taken
    """
    def compute_prob_state_action(self, s1_state_2d, s2_state_2d, action_id):

        # Compute probability of action actually taken
        prob_action = np.ones([self.n_actions])*(float(1-self.prob_trans)/self.n_actions);
        prob_action[action_id] = prob_action[action_id] + self.prob_trans;

        # Probability of reaching s2_state from s1_state if action_id is taken
        prob = 0;

        # Iterate through possible actions
        for i_action in range(len(self.actions)):

            # Get action
            action = self.actions[i_action];
            # Compute new state if this action is executed
            new_state = (s1_state_2d[0] + action[0], s1_state_2d[1] + action[1]);
            # Stay at previous state is new state is out of the grid
            if not self.is_in_grid(new_state):
                new_state = s1_state_2d;

            # If s2_state is reached, add sum probability of the actions that resulted in reaching s2_state
            # Important for the border states
            if new_state == s2_state_2d:
                prob = prob + prob_action[i_action]

        return prob


    """
    Compute the probability of reaching the states from s1_state if action_id is taken
    Shown on a grid representation
    """
    def compute_matrix_proc_state_action(self, s1_state_2d, action_id):

        grid_prob = np.zeros(self.grid.shape);
        for s1_i in range(self.grid.shape[0]):
            for s1_j in range(self.grid.shape[1]):

                grid_prob[s1_i, s1_j] = self.compute_prob_state_action(s1_state_2d, (s1_i, s1_j), action_id);


        return grid_prob;


    ##
    ## Utils function
    ##
    ## Conversion from gridworld representation to standard MDP representation
    ##

    def get_MDP_format(self):

        # Compute the transition probability matrix: n_states x n_states x n_actions
        # P_trans[s1, s2, a] = probability of reaching s2 from s1 when taking action a
        P_trans = np.zeros([self.n_states, self.n_states, self.n_actions]);
        for s1_i in range(self.grid.shape[0]):
            for s1_j in range(self.grid.shape[1]):

                # Compute state index in 1d representation
                s1_state_1d = s1_i*self.grid.shape[1] + s1_j;

                for s2_i in range(self.grid.shape[0]):
                    for s2_j in range(self.grid.shape[1]):
                        # Compute state index in 1d representation
                        s2_state_1d = s2_i*self.grid.shape[1] + s2_j;

                        # Build the transition probability matrix: probability of reaching s2 from s1 when taking action a
                        for i_action in range(self.n_actions):
                            P_trans[s1_state_1d, s2_state_1d, i_action] = self.compute_prob_state_action((s1_i, s1_j), (s2_i, s2_j), i_action);

        # Reward model: reshape the grid reward into 1d array
        rewards = np.reshape(self.grid, self.grid.shape[0]*self.grid.shape[1], order='F');

        # Terminal state into 1d coordinate
        terminal_state_1d = self.terminal_state[0]*self.grid.shape[1] + self.terminal_state[1];

        return self.n_states, self.n_actions, P_trans, rewards, terminal_state_1d;


    """
    Convert 2d representstion (gridworld) into 1d representation (index of the state)
    """
    def convert_state_2d_to_1d(self, state_2d):
        state_1d = state_2d[0]*self.grid.shape[1] + state_2d[1];
        return state_1d;

    """
    Convert 1d representstion (index of the state) into 2d representation (gridworld)
    """
    def convert_state_1d_to_2d(self, state_1d):
        state_2d_i = state_1d // self.grid.shape[1]
        state_2d_j = state_1d % self.grid.shape[1]
        state_2d = (state_2d_i, state_2d_j);
        return state_2d

if __name__ == "__main__":

    print("\n*** Gridworld: Demo ***\n")

    # Create gridworld
    trans_prob = 1;
    size_grid = 10;
    gw = Gridworld(size_grid,trans_prob);

    # Take some actions
    for i in range(30):
        done_flag, st, rw = gw.take_action(2);
        done_flag, st, rw = gw.take_action(3);

    # Grid map of agent's path
    setps = 0;
    grid_path = copy.copy(gw.grid);
    for state_visited in gw.get_state_log():
        grid_path[state_visited[0],state_visited[1]] = setps;
        setps = setps + 1;

    # Plots
    plt.figure()
    plt.subplot(1, 2, 1)
    utils.plot_heatmap(gw.grid, "Reward Map", False)
    plt.subplot(1, 2, 2)
    utils.plot_heatmap(grid_path, "Agent path", False)

    plt.show()









