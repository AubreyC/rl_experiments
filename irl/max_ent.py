# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-09-24 15:11:21
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-09-27 14:47:24

from __future__ import print_function

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


def run_irl_max_ent(n_states, n_actions, p_trans, terminal_state_1d, gamma, feat_states, dem_paths, learning_rate = 0.01, max_path_step=100):

    # Initialize weights of reward
    n_feat = feat_states.shape[1];
    theta = np.random.rand(n_feat,1);

    print('feat_states:{} theta: {}'.format(feat_states.shape, theta.shape))
    # Init reward: Nx1 vector
    rewards = feat_states.dot(theta);



    # Compute empriral feature expectation
    n_dem = len(dem_paths);
    feat_exp = np.zeros([1, n_feat], np,float)
    print("feat_exp 1:{}".format(feat_exp.shape))
    print(feat_states[2, :])

    for path in dem_paths:
        for state_1d in path:
            feat_exp += feat_states[state_1d, :];
    # feat_exp = feat_exp / n_dem;

    # Opimization loop
    n_steps = 20;
    for i_step in range(n_steps):

        # Compute patition function - state visitation
        mu = compute_Z(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma, max_path_step);
        # Compute gradient
        # state_visit: Nx1

        print("feat_exp:{}".format(feat_exp.shape))
        print("mu:{}".format(mu.shape))
        print("feat_states:{}".format(feat_states.shape))

        theta_grad = feat_exp - mu.transpose().dot(feat_states);

        # Update Gradient
        print("theta_grad:{}".format(type(theta_grad)))
        print("theta:{}".format(type(theta)))

        theta = theta + learning_rate*theta_grad.transpose();

        # Compute reward according to new theta
        rewards = feat_states.dot(theta);

        # if i_step % (n_steps/20) == 0:
        print('Step: {}/{}'.format(i_step, n_steps))

    rewards = utils.normalize(rewards);

    return rewards;


"""
Compute partition function
"""
def compute_Z(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma, max_path_step):
    # p_trans[s1, s2, a] = probability of reaching s2 from s1 when taking action a


    print("compute_Z step: {}".format(1))
    # Find optimal policy for current reward
    policy_opt = value_iteration.get_optimal_policy(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma);
    print("compute_Z step: {}".format(2))


    # Compute the state visitation by DP method:
    # a_opt = p_opt(s1)
    # mu(s1,t+1) = sum_s2 mu(s2,t)*P_trans[s2, s1, a_opt]

    print("compute_Z step: {}".format(3))
    # mu(s,t) = visitatio probability of state s at time t
    mu_st = np.zeros([n_states, max_path_step])
    for t in range(-1,max_path_step-1):
        for s1_state in range(n_states):
            for s2_state in range(n_states):
                # for i_action in range(n_actions):
                a_opt = policy_opt[s2_state];
                mu_st[s1_state, t+1] += mu_st[s2_state,t]*p_trans[s2_state, s1_state, a_opt];
                # print('compute_Z loop: s1:{} s2:{} a:{}'.format(s1_state, s2_state, a_opt))
    print("compute_Z step: {}".format(4))


    # mu(s) = sum_t mu(s, t)
    # Sum mu_st along axis 1
    mu = np.sum(mu_st, axis=1);
    mu.shape = (n_states, 1);

    return mu



if __name__ == '__main__':

    print("\n*** Gridworld: Maximum Entropy IRL demo ***\n")

    # Create gridworld
    trans_prob = 0.7;
    size_grid = 10;
    gamma = 0.8;
    gw = gridworld.Gridworld(size_grid, trans_prob);

    # Convert gridwolrd in Finite discrete MDP format
    n_states, n_actions, p_trans, rewards, terminal_state_1d = gw.get_MDP_format();

    # Run Value Iteration algortims
    v_states = value_iteration.run_value_iteration(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma);
    v_states = np.reshape(v_states, gw.grid.shape, order='F');

    # Find optimal policy
    policy_opt = value_iteration.get_optimal_policy(n_states, n_actions, p_trans, rewards, terminal_state_1d, gamma);
    print("Policy: {}".format(policy_opt))

    # Features
    # Reward is a linear combination of features
    # Featurs: NxD with N:n_states et D: n_features
    feat_states = np.identity(n_states);

    # Generate demonstrations:
    N_dem = 100;

    # List of demonstration path
    dem_paths = [];
    for i in range(N_dem):

        # Random start
        state_init_1d = np.random.randint(0, n_states, 1)[0];
        state_init_2d = gw.convert_state_1d_to_2d(state_init_1d);
        gw.reset_agent(state_init_2d);

        # Run the policy
        path_2d = gw.run_policy(policy_opt, max_step=50);

        # Convert state path:
        path_1d = gw.convert_state_log_2d_to_1d(path_2d);

        dem_paths.append(path_1d);


    print("Value:\n", end='');
    for i in range(v_states.shape[0]):
        for j in range(v_states.shape[0]):
            print("%.4f " % (v_states[i,j]), end='')
        print("\n", end='')
    print("\n", end='')


    # # Create a grid to show the optimal policy:
    # # 0: stay, 1: north, 2:east, 3: south, 4: west
    # grid_pol = copy.copy(gw.grid);
    # for s_i in range(grid_pol.shape[0]):
    #     for s_j in range(grid_pol.shape[1]):
    #         s_1d = gw.convert_state_2d_to_1d((s_i, s_j));
    #         a_opt = policy_opt[s_1d];
    #         grid_pol[s_i,s_j] = a_opt;


    # for path_1d in dem_paths:

    #     grid_path = copy.copy(gw.grid);
    #     steps = 0;
    #     for state_1d in path_1d:

    #         state_2d = gw.convert_state_1d_to_2d(state_1d);
    #         grid_path[state_2d[0],state_2d[1]] = steps; #state_visited[0]+state_visited[1];
    #         steps = steps + 1;


    #     plt.figure()
    #     plt.subplot(1, 3, 1)
    #     utils.plot_heatmap(grid_path, "Agent Path", False)
    #     plt.subplot(1, 3, 2)
    #     utils.plot_heatmap(v_states, "Value Function", False)
    #     plt.subplot(1, 3, 3)
    #     utils.plot_policy(grid_pol, gw.actions , "Policy Opt", False)

    #     plt.show()


    # # Run Maximum Entropy IRL
    r_maxent = run_irl_max_ent(n_states, n_actions, p_trans, terminal_state_1d, gamma, feat_states, dem_paths, max_path_step=50);

    r_maxent = np.reshape(r_maxent, (size_grid, size_grid), order='C')
    r_maxent = r_maxent.astype(float)

    print("r_maxent:{}".format(r_maxent));

    # Plot results
    plt.figure()
    plt.subplot(1, 3, 1)
    utils.plot_heatmap(gw.grid, "Original Reward", False)
    plt.subplot(1, 3, 2)
    utils.plot_heatmap(v_states, "Value Function", False)
    plt.subplot(1, 3, 3)
    utils.plot_heatmap(r_maxent, "Recovered Reward: MaxEnt", False)
    plt.show()








