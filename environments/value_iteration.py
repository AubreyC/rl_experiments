# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-08-24 14:56:57
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-08-27 15:12:16
#
# -----------------------------------------
#
# Value Iteration algortihms to find the value function of a Markov Decision Process
# Based on "Reinforcement Learning: An Introduction" Richard S. Sutton and Andrew G. Barto
# Section 4.4
#

import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import utils
import gridworld

"""
Run the value iteration algorithms in the MDP
Ref: "Reinforcement Learning: An Introduction" Richard S. Sutton and Andrew G. Barto section 4.4

inputs:
    n_states           number of states
    n_actions          number of actions
    P_trans            Transition probability matrix: n_states*n_states*n_actions
    rewards            rewards depending only on states in this case: n_statesx1
    terminal_state_1d  terminal state
    discount_f.        discount factor applied to the cummulative reward
    conv_error         convergence criterion
"""
def run_value_iteration(n_states, n_actions, P_trans, rewards, terminal_state_1d, discount_f, conv_error = 0.01):

    # Init Value function with the reward map
    v_states = rewards.copy();

    # Init the error higher than te criterion to start the loop
    error = conv_error + 1;

    # Check convergence criterion
    while error > conv_error:

        # Go over all states
        error = 0;
        for s1_state in range(n_states):

            # If terminal state: stop
            if s1_state == terminal_state_1d:
                continue;

            # Save value function to update it after going through all the states
            values_tmp = v_states.copy()

            # Compute the Q function for all the actions avaible (based on the Bellman operator)
            q_f = []
            for i_action in range(n_actions):
                q = sum(P_trans[s1_state, s2_state, i_action]*(rewards[s1_state] + discount_f*values_tmp[s2_state]) for s2_state in range(n_states));
                q_f.append(q)

            # Value function corresponds to the action that maximizes the cumulative reward
            v_states[s1_state] = max(q_f);

            # Save the largest update of value function, used for convergence criterion
            error = max([error, abs(values_tmp[s1_state] - v_states[s1_state])]);

    return v_states;


if __name__ == '__main__':

    # Create gridworld
    gw = gridworld.Gridworld(10, 0.7);

    # Convert gridwolrd in Finite discrete MDP format
    n_states, n_actions, p_trans, rewards, terminal_state_1d = gw.get_MDP_format();

    # Run Value Iteration algortims
    v_states = run_value_iteration(n_states, n_actions, p_trans, rewards, terminal_state_1d, 0.5);
    v_states = np.reshape(v_states, gw.grid.shape, order='F');

    # Plot results
    plt.figure(figsize=(8,5))
    plt.subplot(1, 2, 1)
    utils.plot_heatmap(gw.grid, "Reward Map", False)
    plt.subplot(1, 2, 2)
    utils.plot_heatmap(v_states, "Value Function", False)
    plt.show()





