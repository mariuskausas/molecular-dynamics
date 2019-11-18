import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_two_cvs(cv1, cv2, bins, weights=None):
    """ 2D histogram two collective variables and prepare input for FES plot.

    Weights can be provided to weight the resulted 2D histogram. 
    cv1, cv2 and weights should be provided as (n,) numpy arrays.
    """
    
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

    return -np.log(H.T), xedges, yedges


def plot_2D_fes(H, xedges, yedges, output_name):
    """ Helper plotting function."""
    
    plt.figure(figsize=(8, 7))
    plt.imshow(H, interpolation='nearest', origin='low',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect="auto", cmap="Spectral")
    plt.savefig(output_name + ".png", dpi=600)
    plt.colorbar()
    plt.show()


def two_variable_fes(path_to_cv1, path_to_cv2, bins, output_name, weights=None):
    """ Plot 2D FES."""

    # Check if the weights are provided
    if weights is not None:
        weights = load_weights(weights)

    H, xedges, yedges = convert_fes(cv1, cv2, bins, weights)
    plot_2Dfes(H, xedges, yedges, output_name)

    return

