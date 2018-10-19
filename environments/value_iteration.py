# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-08-24 14:56:57
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-09-27 14:28:08
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
import copy

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
def run_value_iteration(n_states, n_actions, P_trans, rewards, terminal_state_1d, discount_f, conv_error = 1e-9):

    # Init Value function with the reward map
    #v_states = rewards.copy();
    v_states = np.zeros([n_states]);

    # Check convergence criterion
    while True:

        for s1_state in range(n_states):

            # Save value function to update it after going through all the states
            values_tmp = v_states.copy()

            # Compute the Q function for all the actions avaible (based on the Bellman operator)
            q_f = []
            for i_action in range(n_actions):
                q = sum(P_trans[s1_state, s2_state, i_action]*(rewards[s2_state] + discount_f*values_tmp[s2_state]) for s2_state in range(n_states));
                q_f.append(q)

            # Value function corresponds to the action that maximizes the cumulative reward
            v_states[s1_state] = max(q_f);

        # Convergence criterion
        error = max( [abs(values_tmp[s1_state] - v_states[s1_state]) for s1_state in range(n_states)]);
        if error < conv_error:
            break;

    return v_states;


def get_optimal_policy(n_states, n_actions, P_trans, rewards, terminal_state_1d, gamma):

    # Get the Value function with Value Iteration algorithms
    v_states = run_value_iteration(n_states, n_actions, P_trans, rewards, terminal_state_1d, gamma);

    policy_opt = np.zeros(n_states, np.int);

    # Go through each state
    for s1_state in range(n_states):

        # Compute the Q function for each state based on the value function of the next states
        q_f = np.zeros(n_actions);
        for i_action in range(n_actions):
            q = sum(P_trans[s1_state, s2_state, i_action]*(rewards[s1_state] + gamma*v_states[s2_state]) for s2_state in range(n_states));
            q_f[i_action] = q;

        # Chosing the action that maximize the value function (cumulative reward)
        a_opt = np.argmax(q_f);

        policy_opt[s1_state] = int(a_opt);

    return policy_opt;

if __name__ == '__main__':

    print("\n*** Gridworld: Value Iteration demo ***\n")

    # Create gridworld
    trans_prob = 0.8;
    size_grid = 10;
    gamma = 0.7;
    gw = gridworld.Gridworld(size_grid, trans_prob);

    # Convert gridwolrd in Finite discrete MDP format
    n_states, n_actions, p_trans, rewards, terminal_state_1d = gw.get_MDP_format();

    # Run Value Iteration algortims
    v_states = run_value_iteration(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma);
    v_states = np.reshape(v_states, gw.grid.shape, order='F');

    # Find aptimal policy
    policy_opt = get_optimal_policy(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma);

    # Run an agent with the optimal policy
    # It's interesting to note that the optimal policy "chose" to go all the way to the right side and down
    # This is due to the argmax function taking the first occurence of the maximum Q_function value when chosing the action
    # Due to the order of actions, it choses to go right (even if goind south would be as optimal)
    while (not gw.is_over(gw.get_current_state())):
        s_1d = gw.convert_state_2d_to_1d(gw.get_current_state());
        a_opt = policy_opt[s_1d];
        done_flag, st, rw = gw.take_action(a_opt);

    # Create a grid to show the optimal policy:
    # 0: stay, 1: north, 2:east, 3: south, 4: west
    grid_pol = copy.copy(gw.grid);
    for s_i in range(grid_pol.shape[0]):
        for s_j in range(grid_pol.shape[1]):
            s_1d = gw.convert_state_2d_to_1d((s_i, s_j));
            a_opt = policy_opt[s_1d];
            grid_pol[s_i,s_j] = a_opt;

    # Create a Grid to show the path of the worker
    # The number of steps of the agent is shwon
    grid_path = copy.copy(gw.grid);
    steps = 0;
    for state_visited in gw.get_state_log():
        grid_path[state_visited[0],state_visited[1]] = steps; #state_visited[0]+state_visited[1];
        steps = steps + 1;

    # Plot results
    plt.figure()
    # plt.subplot(1, 2, 1)
    utils.plot_heatmap(gw.grid, "Reward Map", False, )

    plt.figure()
    # plt.subplot(1, 2, 2)
    utils.plot_heatmap(v_states, "Value Function", False, True)

    plt.figure()
    # plt.subplot(1, 2, 1)
    utils.plot_heatmap(grid_path, "Agent Path", False, False)

    plt.figure()
    # plt.subplot(1, 2, 2)
    utils.plot_policy(grid_pol, gw.actions , "Policy Opt", False)

    # utils.plot_heatmap(grid_pol, "Policy Opt", False)

    plt.show()





