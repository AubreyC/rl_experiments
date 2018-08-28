# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-08-27 15:26:27
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-08-28 17:02:09
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
from scipy.optimize import linprog
import sys

from cvxopt import matrix, solvers


sys.path.append("../environments")

import utils
import gridworld
import value_iteration



def lp_irl(n_states, n_actions, P_trans, gamma, policy_opt, lambda_1=10, R_MAX = 10):

    # Inspired from: http://matthewja.com/pdfs/irl.pdf

    # Solving the system:
    # min c^T X
    # st AX <= b
    #
    # Tricks: dummy vector: X = [R, M, u]

    # Define c
    c = np.zeros(n_states*3);
    c[n_states:2*n_states] = 1;
    c[n_states:] = -lambda_1;

    # Define A:
    A = np.zeros([2*n_states*n_actions + 2*n_states + 2*n_states, n_states*3]);

    # Define b:
    b = np.zeros([2*n_states*n_actions + 2*n_states + 2*n_states]);


    for a in range(n_actions):

        # Constructing P_a_star matrix: Each row (state) have the row of the Transition Probability Matrix with the optimal action
        P_a_star = np.zeros([n_states, n_states]);
        for s1_state in range(n_states):
            a_star = policy_opt[s1_state];
            P_a_star[s1_state, :] = P_trans[s1_state, :, a_star];

        print("P_a_star: {}".format(P_a_star))
        tmp = - (P_a_star - P_trans[:, :, a]).dot(np.linalg.inv(np.identity(n_states) - gamma*P_a_star))


        A[a*2*n_states: a*2*n_states + n_states, :n_states] = tmp;
        A[a*2*n_states: a*2*n_states + n_states, n_states : 2*n_states] = np.identity(n_states);
        A[a*2*n_states: a*2*n_states + n_states, 2*n_states : 3*n_states] = np.zeros([n_states,n_states]);

        A[a*2*n_states + n_states : a*2*n_states + 2*n_states, :n_states] = tmp;
        A[a*2*n_states + n_states : a*2*n_states + 2*n_states, n_states : 2*n_states] = np.zeros([n_states,n_states]);
        A[a*2*n_states + n_states : a*2*n_states + 2*n_states, 2*n_states : 3*n_states] = np.zeros([n_states,n_states]);

    A[n_actions*2*n_states: n_actions*2*n_states + n_states, :n_states] = -np.identity(n_states);
    A[n_actions*2*n_states: n_actions*2*n_states + n_states, n_states : 2*n_states] = np.zeros([n_states,n_states]);
    A[n_actions*2*n_states: n_actions*2*n_states + n_states, 2*n_states : 3*n_states] = -np.identity(n_states);

    A[n_actions*2*n_states + n_states : n_actions*2*n_states + 2*n_states, :n_states] = +np.identity(n_states);
    A[n_actions*2*n_states + n_states : n_actions*2*n_states + 2*n_states, n_states : 2*n_states] = np.zeros([n_states,n_states]);
    A[n_actions*2*n_states + n_states : n_actions*2*n_states + 2*n_states, 2*n_states : 3*n_states] = -np.identity(n_states);


    # Add this to boudn R
    # -I R <= Rmax 1
    # I R <= Rmax 1
    A[n_actions*2*n_states + 2*n_states : n_actions*2*n_states + 3*n_states, :n_states] = +np.identity(n_states);
    A[n_actions*2*n_states + 2*n_states : n_actions*2*n_states + 3*n_states, n_states : 2*n_states] = np.zeros([n_states,n_states]);
    A[n_actions*2*n_states + 2*n_states : n_actions*2*n_states + 3*n_states, 2*n_states : 3*n_states] = np.zeros([n_states,n_states]);

    A[n_actions*2*n_states + 3*n_states : n_actions*2*n_states + 4*n_states, :n_states] = -np.identity(n_states);
    A[n_actions*2*n_states + 3*n_states : n_actions*2*n_states + 4*n_states, n_states : 2*n_states] = np.zeros([n_states,n_states]);
    A[n_actions*2*n_states + 3*n_states : n_actions*2*n_states + 4*n_states, 2*n_states : 3*n_states] = np.zeros([n_states,n_states]);

    b[n_actions*2*n_states + 2*n_states : n_actions*2*n_states + 3*n_states] = R_MAX;
    b[n_actions*2*n_states + 3*n_states : n_actions*2*n_states + 4*n_states] = R_MAX;

    # # Solve the LP system
    # res = linprog(c, A_ub=A, b_ub=b, options={"disp": True})

    # R_lp = res['x'][:n_states];

    # return R_lp

    sol = solvers.lp(matrix(c), matrix(A), matrix(b))
    rewards = sol['x'][:n_states]
    # rewards = normalize(rewards) * R_max
    return rewards

if __name__ == '__main__':

    print("\n*** Gridworld: Value Iteration demo ***\n")

    # Create gridworld
    trans_prob = 0.8;
    size_grid = 4;
    gamma = 0.8;
    gw = gridworld.Gridworld(size_grid, trans_prob);

    # Convert gridwolrd in Finite discrete MDP format
    n_states, n_actions, p_trans, rewards, terminal_state_1d = gw.get_MDP_format();

    # Run Value Iteration algortims
    v_states = value_iteration.run_value_iteration(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma);
    v_states = np.reshape(v_states, gw.grid.shape, order='F');

    # Find aptimal policy
    policy_opt = value_iteration.get_optimal_policy(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma);



    R_lp = lp_irl(n_states, n_actions, p_trans, gamma, policy_opt);
    print(R_lp)
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
    plt.subplot(1, 2, 1)
    utils.plot_heatmap(gw.grid, "Reward Map", False)
    plt.subplot(1, 2, 2)
    utils.plot_heatmap(v_states, "Value Function", False)

    plt.figure()
    plt.subplot(1, 2, 1)
    utils.plot_heatmap(grid_path, "Agent Path", False)
    plt.subplot(1, 2, 2)
    utils.plot_heatmap(grid_pol, "Policy Opt", False)

    plt.show()