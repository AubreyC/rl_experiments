# -*- coding: utf-8 -*-
# @Author: Aubrey
# @Date:   2018-08-24 13:59:33
# @Last Modified by:   Aubrey
# @Last Modified time: 2018-08-24 18:50:26

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""
Draw heatmap: Inspired from https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
"""
def plot_heatmap(grid, title, show_flag = False):


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
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            text = plt.text(j, i, '{0:.2f}'.format(grid[i, j]),
                           ha="center", va="center", color="w")

    plt.title(title)
    plt.tight_layout()

    if show_flag:
        plt.show()