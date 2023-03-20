"""
Utilities for visualizing and analyzing 3D simplex plots in multi-class
probability space.

This module provides functions for creating 3D simplex plots, configuring their
appearance, and plotting decision regions based on the idea of a
NearestNeighbors object fitted on the simplex space (choosing the search
neighbors to be the the edge of the simplex is equivalent to argmax hard
predictions).
"""
# Note: to deal with Axes3D I had to use mostly theese references:
# https://www.oreilly.com/library/view/python-data-science/9781491912126/ch04.html
# https://matplotlib.org/3.5.0/tutorials/toolkits/mplot3d.html#mpl_toolkits.mplot3d.Axes3D.plot_trisurf


import numpy as np
import matplotlib.pyplot as plt


def plot_simplex(ax_3d=None, grid_resolution=221, color='k', alpha=0.25):
    """Plots a 3-simplex using trisurf on a 3D axis.

    Parameters
    ----------
    ax_3d : matplotlib.axes._subplots.Axes3DSubplot or None, default=None
        The 3D axis on which the simplex will be plotted. If not provided, a
        new 3D axis will be created. If `None`, then the function will create
        the 3D axis.

    grid_resolution : int, default=221
        Number of points used for creating the linspace. Default is 221.

    color : str or tuple, default="k"
        The color of the simplex. Default is black.

    alpha : float, default=0.25
        The transparency of the simplex.

    Returns
    -------
    ax_3d : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis containing the plotted simplex.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> ax = plt.figure().add_subplot(projection='3d')
    >>> ax = plot_simplex(ax)
    >>> plt.show()
    """
    if ax_3d is None:
        ax_3d = plt.figure().add_subplot(projection='3d')

    p1 = np.linspace(0, 1, grid_resolution)
    p2 = np.linspace(0, 1, grid_resolution)
    P1, P2 = np.meshgrid(p1, p2)
    P3 = 1 - P1 - P2

    mask_simplex = (P1 >= 0) & (P2 >= 0) & (P3 >= 0)
    P1, P2, P3 = P1[mask_simplex], P2[mask_simplex], P3[mask_simplex]
    ax_3d.plot_trisurf(P1, P2, P3, color=color, alpha=alpha)

    return ax_3d


def clean_simplex_ax(ax_3d):
    """Configures the appearance of a 3D axis containing a simplex plot.

    Parameters
    ----------
    ax_3d : matplotlib.axes._subplots.Axes3DSubplot
        The 3D axis containing the simplex plot.

    Returns
    -------
    ax_3d : matplotlib.axes._subplots.Axes3DSubplot
        The cleaned and configured 3D axis containing the simplex plot.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> ax_3d = plot_simplex()
    >>> ax_3d = clean_simplex_ax(ax_3d)
    >>> plt.show()
    """
    ax_3d.set_xlim([0, 1])
    ax_3d.set_ylim([0, 1])
    ax_3d.set_zlim([0, 1])

    ax_3d.tick_params(axis='both', which='major', pad=-4)
    ticks = np.linspace(0, 1, 5)
    ticks_str = [str(x) if x not in [0, 1] else str(int(x)) for x in ticks]

    fontsize = 6
    ax_3d.set_xticks(ticks)
    ax_3d.set_xticklabels(ticks_str, fontsize=fontsize)
    ax_3d.set_yticks(ticks)
    ax_3d.set_yticklabels(ticks_str, fontsize=fontsize)
    ax_3d.set_zticks(ticks)
    ax_3d.set_zticklabels(['0'] + [''] * 3 + ['1'], fontsize=fontsize)

    ax_3d.set_xlabel('Class 0 prob', labelpad=-8)
    ax_3d.set_ylabel('Class 1 prob', labelpad=-8)
    ax_3d.set_zlabel('Class 2 prob', labelpad=-12)

    ax_3d.view_init(elev=45, azim=45, roll=0)

    return ax_3d


def plot_3d_regions_over_simplex(
    fitted_nn, ax_3d=None, grid_resolution=221, colors=None, alpha=0.25
):
    """Plots 3D decision regions for a fitted NearestNeighbors object on a 3D
    axis over the simplex region.

    Parameters
    ----------
    fitted_nn : sklearn.neighbors.NearestNeighbors
        A fitted NearestNeighbors object.

    ax_3d : matplotlib.axes._subplots.Axes3DSubplot or None, default=None
        The 3D axis on which to plot the decision regions. If `None`, then the
        function will create the 3D axis.

    grid_resolution : int, default=221
        Number of points used for creating the linspace.

    colors : list of str or tuple or None, default=None
        A list of colors for the decision regions. If `None`, then the colors
        will be red, blue and green.

    alpha : float, default=0.25
        The transparency of the decision regions.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from sklearn.neighbors import NearestNeighbors
    >>> # Create a NearestNeighbors object and fit it with some data
    >>> nn = NearestNeighbors(n_neighbors=1)
    >>> X = np.array([[0.33, 0.33, 0.34], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2]])
    >>> nn.fit(X)
    >>> ax_3d = plot_3d_regions(nn)
    >>> ax_3d = clean_simplex_ax(ax_3d)
    >>> plt.show()
    """
    if ax_3d is None:
        ax_3d = plt.figure().add_subplot(projection='3d')

    if colors is None:
        colors = ['r', 'b', 'g']

    p1 = np.linspace(0, 1, grid_resolution)
    p2 = np.linspace(0, 1, grid_resolution)
    P1, P2 = np.meshgrid(p1, p2)
    P3 = 1 - P1 - P2

    mask_simplex = (P1 >= 0) & (P2 >= 0) & (P3 >= 0)
    P1, P2, P3 = P1[mask_simplex], P2[mask_simplex], P3[mask_simplex]

    _, ind = fitted_nn.kneighbors(
        np.vstack([np.ravel(P1), np.ravel(P2), np.ravel(P3)]).T
    )

    for i, color in enumerate(colors):
        mask = ind[:, 0] == i
        ax_3d.plot_trisurf(
            P1[mask], P2[mask], P3[mask], color=color, alpha=alpha
        )

    return ax_3d
