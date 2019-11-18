d = """
=============================================
Generate a 2D free energy surface of two collective variables as a contour plot.
In addition, overlay a count histogram for a third collective variable.

Before use:
- Install numpy and matplotlib

Marius Kausas					   2019 11 18
=============================================
"""

import numpy as np
import matplotlib.pyplot as plt


def prepare_three_cvs(cv1, cv2, cv3, bins, weights):
    """ Prepare input for FES (2 CVs) and Contour(1 CV) plot.

    Weights can be provided to weight the resulted 2D histogram. 
    cv1, cv2, cv3 and weights should be provided as (n,) numpy arrays.
    """
    
    ## Generate a 2D histogram of first two collective variables
    # Define range of collective variables
    xmin = np.floor(cv1.min())
    xmax = np.ceil(cv1.max())
    ymin = np.floor(cv2.min())
    ymax = np.ceil(cv2.max())

    # Calculate bin size for each collective variable
    xbin_size = (xmax - xmin) / bins
    ybin_size = (ymax - ymin) / bins
    xedges = np.arange(xmin, xmax, xbin_size)
    yedges = np.arange(ymin, ymax, ybin_size)

    # Histogram values into 2D array
    H, xedges, yedges = np.histogram2d(cv1, cv2, bins=(xedges, yedges), weights=weights, normed=True)
    
    ## Generate a 2D histogram for the third collective variable
    # Stack collective variables
    stacked_cvs = np.vstack((cv1, cv2, cv3)).T

    # Create zeros arrays
    value_mat = np.zeros(H.shape)
    density_mat = np.zeros(H.shape)

    # Calculate a normalized count matrix of a third collective variable
    for indx, row in enumerate(stacked_cvs):
        scv1 = row[0]
        scv2 = row[1]
        scv3 = row[2]

	# Bin positions for indexing
        scv1_bin_indx = np.where((xedges <= scv1) == True)[0][-1] - 1
        scv2_bin_indx = np.where((yedges <= scv2) == True)[0][-1] - 1

        value_mat[scv1_bin_indx, scv2_bin_indx] += scv3
        density_mat[scv1_bin_indx, scv2_bin_indx] += 1

    J = np.divide(value_mat, density_mat)

    return -np.log(H.T), J.T, xedges, yedges


def plot_2D_contour_count(H, J, xedges, yedges, output_name):
    """ Helper plotting function."""
    
    fs = 22
    fig = plt.figure(figsize=[6, 4])
    ax = fig.add_subplot(111)
    
    imshow = ax.imshow(J, interpolation='nearest', origin='low',
		       extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect="auto", cmap="PRGn")
    fig.colorbar(imshow, ax = ax)
    
    contour = ax.contour(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap="Spectral", levels=10, linewidths=1)
    fig.colorbar(contour, ax = ax)
    
    fig.tight_layout()
    plt.savefig(output_name + ".png", dpi=300)
    plt.show()


def three_variable_contour_count(cv1, cv2, cv3, bins, output_name, weights=None):
    """ Plot a 2D contour FES plot with an overlay of count matrix."""

    # Check if the weights are provided
    if weights is not None:
        weights = load_weights(weights)

    H, J, xedges, yedges = prepare_three_cvs(cv1, cv2, cv3, bins, weights=weights)
    plot_2D_contour_count(H, J, xedges, yedges, output_name)

    return
