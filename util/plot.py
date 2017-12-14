#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Utility functions for plotting.

Created on Fri Mar 24 14:27:16 2017

@author: jje
"""

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

def plot_wireframe( data, legend_label="_nolabel_", figno=None ):
    """Make and label a wireframe plot.

Parameters:
    data : dict
        key   : "x","y","z"
        value : tuple (rank-2 array in meshgrid format, axis label)

Return value:
    ax
        The Axes3D object that was used for plotting.
"""
    # http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
    fig = plt.figure(figno)

    # Axes3D has a tendency to underestimate how much space it needs; it draws its labels
    # outside the window area in certain orientations.
    #
    # This causes the labels to be clipped, which looks bad. We prevent this by creating the axes
    # in a slightly smaller rect (leaving a margin). This way the labels will show - outside the Axes3D,
    # but still inside the figure window.
    #
    # The final touch is to set the window background to a matching white, so that the
    # background of the figure appears uniform.
    #
    fig.patch.set_color( (1,1,1) )
    fig.patch.set_alpha( 1.0 )
    x0y0wh = [ 0.02, 0.02, 0.96, 0.96 ]  # left, bottom, width, height      (here as fraction of subplot area)

    ax = mpl_toolkits.mplot3d.axes3d.Axes3D(fig, rect=x0y0wh)

    X,xlabel = data["x"]
    Y,ylabel = data["y"]
    Z,zlabel = data["z"]
    ax.plot_wireframe( X, Y, Z, label=legend_label )

#    ax.view_init(34, 140)
#    ax.view_init(34, -40)
    ax.view_init(34, -130)
    ax.axis('tight')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_title(zlabel)

    return ax
