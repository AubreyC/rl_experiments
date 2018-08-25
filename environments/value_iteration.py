# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-08-24 14:56:57
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-08-25 13:03:52
#
# -----------------------------------------
#
# Value Iteration algortihms to find the value function of a Markov Decision Process
# Based on "Reinforcement Learning: An Introduction" Richard S. Sutton and Andrew G. Barto
# Section 4.4
#

import math
import gridworld
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import utils
"""
Run the value iteration algorithms in the MDP

inputs:
    n_states       number of states
    n_actions      number of actions
    P_trans        Transition probability matrix: n_states*n_states*n_actions
    rewards        rewards depending only on states in this case: n_statesx1
"""
def run_value_iteration(n_states, n_actions, P_trans, rewards, discount_f, conv_error = 0.01):

    v_states = np.zeros(n_states);
    v_states[n_states-1] = rewards[n_states-1];

    error = conv_error + 1;
    while error > conv_error:

        error = 0;
        for s1_state in range(n_states):


            if s1_state == n_states - 1:
                continue;

            values_tmp = v_states.copy()

            q_f = []
            for i_action in range(n_actions):
                q = sum(P_trans[s1_state, s2_state, i_action]*(rewards[s1_state] + discount_f*values_tmp[s2_state]) for s2_state in range(n_states));
                q_f.append(q)

            v_states[s1_state] = max(q_f);

            error = max([error, abs(values_tmp[s1_state] - v_states[s1_state])]);

    return v_states;


if __name__ == '__main__':

    # Create gridworld
    gw = gridworld.Gridworld(10, 0.7);
    n_states, n_actions, p_trans, rewards = gw.get_MDP_format();

    v_states = run_value_iteration(n_states, n_actions, p_trans, rewards, 0.5);

    v_states = np.reshape(v_states, gw.grid.shape, order='F');


    plt.figure()
    # plt.subplot(1, 2, 1)
    # utils.plot_heatmap(gw.grid, "Reward Map", False)
    # plt.subplot(1, 2, 2)
    utils.plot_heatmap(v_states, "Value Function", False)

    plt.show()

    # Take some actions


    # plt.show()



