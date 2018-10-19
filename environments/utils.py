# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-08-24 13:59:33
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-09-27 18:07:35
from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np


def print_value(v_states):
    print("Value:\n", end='');
    for i in range(v_states.shape[0]):
        for j in range(v_states.shape[0]):
            print("%.4f " % (v_states[i,j]), end='')
        print("\n", end='')
    print("\n", end='')

def print_policy(grid_policy, actions):

    # Loop over data dimensions and create text annotations.
    for i in range(grid_policy.shape[0]):
        for j in range(grid_policy.shape[1]):

            act_str = "";
            if actions[int(grid_policy[i, j])] == (0,1):
                act_str = ">";

            elif actions[int(grid_policy[i, j])] == (0,-1):
                act_str = "<";

            elif actions[int(grid_policy[i, j])] == (1,0):
                act_str = "v";

            elif actions[int(grid_policy[i, j])] == (0,-1):
                act_str = "^";

            elif actions[int(grid_policy[i, j])] == (0,0):
                act_str = "o";

            print("%s " % (act_str), end='');

        print("\n", end='')

"""
Draw heatmap: Inspired from https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
"""
def plot_heatmap(grid, title, show_flag = False, show_numbers = True):


    # # We want to show all ticks...
    # ax.set_xticks(np.arange(len(farmers)))
    # ax.set_yticks(np.arange(len(vegetables)))
    # # ... and label them with the respective list entries
    # ax.set_xticklabels(farmers)
    # ax.set_yticklabels(vegetables)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    plt.imshow(grid)

    # Loop over data dimensions and create text annotations.
    if(show_numbers):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                text = plt.text(j, i, '{0:.1f}'.format(grid[i, j]),
                               ha="center", va="center", color="black")

    plt.title(title)
    plt.tight_layout()

    if show_flag:
        plt.show()

def plot_policy(grid, actions, title, show_flag = False, show_numbers = True):

    # grid_null = np.zeros(grid.shape)
    # plt.imshow(grid_null)

    plt.imshow(grid)

    # Loop over data dimensions and create text annotations.
    if(show_numbers):
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):

                act_str = "";
                if actions[int(grid[i, j])] == (0,1):
                    act_str = ">";

                elif actions[int(grid[i, j])] == (0,-1):
                    act_str = "<";

                elif actions[int(grid[i, j])] == (1,0):
                    act_str = "v";

                elif actions[int(grid[i, j])] == (0,-1):
                    act_str = "^";

                elif actions[int(grid[i, j])] == (0,0):
                    act_str = "o";

                text = plt.text(j, i, '{}'.format(act_str),
                               ha="center", va="center", color="w")

    plt.title(title)
    plt.tight_layout()

    if show_flag:
        plt.show()

def normalize(values):
  """
  normalize to (0, max_val)
  input:
    vals: 1d array
  """
  min_v = np.min(values)
  max_v = np.max(values)
  return (values - min_v) / (max_v - min_v)
