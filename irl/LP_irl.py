# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-08-27 15:26:27
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-09-24 16:31:56
#
# -----------------------------------------
#
# Linear Programming formulation of Inverse Reinforcement Learning
# Finite and Discrete MDP
#

import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
# from scipy.optimize import linprog
import sys

from cvxopt import matrix, solvers


sys.path.append("../environments")

import utils
import gridworld
import value_iteration

def run_irl_lp(n_states, n_actions, P_trans, gamma, policy_opt, lambda_1=10, R_MAX = 10):

    # Inspired from: http://matthewja.com/pdfs/irl.pdf

    # Solving the system:
    # min c^T X
    # st AX <= b
    #
    # Tricks: dummy vector: X = [R, M, u]

    # Define c
    c = np.zeros(n_states*3);
    c[n_states:2*n_states] = -1;
    c[2*n_states:] = lambda_1;

    # Define A:
    A = np.zeros([2*n_states*(n_actions-1) + 2*n_states + 2*n_states, n_states*3]);

    # Define b:
    b = np.zeros([2*n_states*(n_actions-1) + 2*n_states + 2*n_states]);


    for s1_state in range(n_states):
        a_opt = int(policy_opt[s1_state]);

        cnt_a = 0;
        for a in range(n_actions):
            if(not a == a_opt):
                tmp = - (P_trans[s1_state, :, a_opt] - P_trans[s1_state, :, a]).dot(np.linalg.inv(np.identity(n_states) - gamma*P_trans[:,:,a_opt]))

                A[2*s1_state*(n_actions-1) + 2*cnt_a, :n_states] = tmp
                A[2*s1_state*(n_actions-1) + 2*cnt_a, n_states + s1_state] = 1;

                A[2*s1_state*(n_actions-1) + 2*cnt_a + 1 , :n_states] = tmp
                A[2*s1_state*(n_actions-1) + 2*cnt_a + 1 , n_states + s1_state] = 0;

                cnt_a = cnt_a + 1;


    A[2*n_states*(n_actions-1): 2*n_states*(n_actions-1) + n_states, :n_states] = -np.identity(n_states);
    A[2*n_states*(n_actions-1): 2*n_states*(n_actions-1) + n_states,  n_states : 2*n_states] = np.zeros([n_states,n_states]);
    A[2*n_states*(n_actions-1): 2*n_states*(n_actions-1) + n_states,  2*n_states : 3*n_states] = -np.identity(n_states);

    A[2*n_states*(n_actions-1) + n_states : 2*n_states*(n_actions-1) + 2*n_states , :n_states] = +np.identity(n_states);
    A[2*n_states*(n_actions-1) + n_states : 2*n_states*(n_actions-1) + 2*n_states , n_states : 2*n_states] = np.zeros([n_states,n_states]);
    A[2*n_states*(n_actions-1) + n_states : 2*n_states*(n_actions-1) + 2*n_states , 2*n_states : 3*n_states] = -np.identity(n_states);

    # # Add this to boudn R
    # # -I R <= Rmax 1
    # # I R <= Rmax 1

    A[2*n_states*(n_actions-1) + 2*n_states : 2*n_states*(n_actions-1) + 3*n_states , :n_states] = -np.identity(n_states);
    A[2*n_states*(n_actions-1) + 2*n_states : 2*n_states*(n_actions-1) + 3*n_states , n_states : 2*n_states] = np.zeros([n_states,n_states]);
    A[2*n_states*(n_actions-1) + 2*n_states : 2*n_states*(n_actions-1) + 3*n_states , 2*n_states : 3*n_states] = np.zeros([n_states,n_states]);

    A[2*n_states*(n_actions-1) + 3*n_states : 2*n_states*(n_actions-1) + 4*n_states , :n_states] = np.identity(n_states);
    A[2*n_states*(n_actions-1) + 3*n_states : 2*n_states*(n_actions-1) + 4*n_states , n_states : 2*n_states] = np.zeros([n_states,n_states]);
    A[2*n_states*(n_actions-1) + 3*n_states : 2*n_states*(n_actions-1) + 4*n_states , 2*n_states : 3*n_states] = np.zeros([n_states,n_states]);

    b[2*n_states*(n_actions-1) + 2*n_states : 2*n_states*(n_actions-1) + 4*n_states] = R_MAX;

    sol = solvers.lp(matrix(c), matrix(A), matrix(b))
    rewards = sol['x'][:n_states]
    rewards = utils.normalize(rewards) * R_MAX

    return rewards

if __name__ == '__main__':

    print("\n*** Gridworld: Value Iteration demo ***\n")

    # Create gridworld
    trans_prob = 0.7;
    size_grid = 10;
    gamma = 0.5;
    gw = gridworld.Gridworld(size_grid, trans_prob);

    # Convert gridwolrd in Finite discrete MDP format
    n_states, n_actions, p_trans, rewards, terminal_state_1d = gw.get_MDP_format();

    # Run Value Iteration algortims
    v_states = value_iteration.run_value_iteration(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma);
    v_states = np.reshape(v_states, gw.grid.shape, order='F');

    # Find aptimal policy
    policy_opt = value_iteration.get_optimal_policy(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma);

    print("Policy: {}".format(policy_opt))
    R_lp = run_irl_lp(n_states, n_actions, p_trans, gamma, policy_opt);
    R_lp = np.reshape(R_lp, (size_grid, size_grid), order='C')
    print("Reward found:\n{}\n".format(R_lp))

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
    plt.subplot(1, 3, 1)
    utils.plot_heatmap(gw.grid, "Original Reward", False)
    plt.subplot(1, 3, 2)
    utils.plot_heatmap(v_states, "Value Function", False)
    plt.subplot(1, 3, 3)
    utils.plot_heatmap(R_lp, "Recovered Reward", False)

    plt.figure()
    plt.subplot(1, 2, 1)
    utils.plot_heatmap(grid_path, "Agent Path", False)
    plt.subplot(1, 2, 2)
    utils.plot_policy(grid_pol, gw.actions , "Policy Opt", False)

    plt.figure()
    utils.plot_heatmap(R_lp, "Reward recovered", False)

    plt.show()
