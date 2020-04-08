d = """
=============================================
Generate a 2D free energy surface of two collective variables.

Before use:
- Install numpy and matplotlib

Marius Kausas					   2020 04 08
=============================================
"""


import numpy as np
import matplotlib.pyplot as plt


def get_histogram(xval, yval, nbins, weights):
    """
    Compute 2D histogram of two variables.

    Parameters
    ----------
    xval : ndarray
        Numpy array (N, ) of the first variable.
    yval : ndarray
        Numpy array (N, ) of the first variable.
    nbins : int
        Number of bins.
    weights : ndarray
        Numpy array (N, ) of weights.

    Returns
    -------
    z : ndarray
        Numpy array (N, N) of the histogram.
    xedges : ndarray
        Numpy array (N, ) of the x bin edges.
    yedges : ndarray
        Numpy array (N, ) of the y bin edges.
    """

    z, x, y = np.histogram2d(xval, yval, bins=nbins, weights=weights)
    xedges = 0.5 * (x[:-1] + x[1:])
    yedges = 0.5 * (y[:-1] + y[1:])

    return z.T, xedges, yedges


def density(z):
    """
    Compute a probability density function.

    Parameters
    ----------
    z : ndarray
        Numpy array (N, N) of the histogram.

    Returns
    -------
    out : ndarray
        Numpy array (N, N) of the probability density function
    """

    return z / float(z.sum())


def free_energy(z):
    """
    Transform probability density function into free energy.

    Parameters
    ----------
    z : ndarray
        Numpy array (N, N) of the probability density function

    Returns
    -------
    out : ndarray
        Numpy array (N, N) of the free energy surface values.
    """

    prob = density(z)
    free_energy = np.inf * np.ones(shape=z.shape)
    nonzero = prob.nonzero()
    free_energy[nonzero] = -np.log(prob[nonzero])

    return free_energy


def prepare_two_cvs(path_to_cv1, path_to_cv2, nbins, weights):
    """
    Convert two collective variables into a free energy surface.

    Parameters
    ----------
    path_to_cv1 : str
        Path to the file containing first collective variable values.
    path_to_cv2 : str
        Path to the file containing second collective variable values.
    nbins : int
        Number of bins.
    weights : ndarray
        Numpy array (N, ) of weights.

    Returns
    -------
    f : ndarray
        Numpy array (N, N) of the free energy surface values.
    xedges : ndarray
        Numpy array (N, ) of the x bin edges.
    yedges : ndarray
        Numpy array (N, ) of the y bin edges.
    """

    # Load collective variables
    cv1 = np.loadtxt(path_to_cv1)
    cv2 = np.loadtxt(path_to_cv2)

    # Histogram values into 2D array
    H, xedges, yedges = get_histogram(cv1, cv2, nbins, weights)
    f = free_energy(H)

    return f, xedges, yedges


def plot_2D_contour(f, xedges, yedges, cv1_label, cv2_label, output_name):
    """ Plotting function for 2D free energy surface."""

    fs = 22
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111)

    # Normalize free energy surface to 0
    f -= f.min()
    # Extract maximal FES value for contour level plotting
    m = f[np.isfinite(f)].max()

    # Filled contour plot
    contour = ax.contourf(f, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                          cmap="Spectral",
                          levels=np.linspace(0, np.round(m), np.round(m) * 2 + 1),
                          corner_mask=True,
                          alpha=0.75)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.set_ylabel("$-\ln P$ MD", fontsize=fs)

    # Contour outline plot
    contour = ax.contour(f, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                         cmap="gist_gray",
                         levels=np.linspace(0, np.round(m), np.round(m) * 2 + 1),
                         linewidths=1,
                         corner_mask=True,
                         alpha=1)

    ax.tick_params(labelsize=fs)

    ax.set_xlabel(cv1_label, fontsize=fs)
    ax.set_ylabel(cv2_label, fontsize=fs)

    fig.tight_layout()
    plt.savefig(output_name + ".png", dpi=300)
    plt.close()


def two_variable_fes(path_to_cv1, path_to_cv2, nbins, cv1_label, cv2_label, output_name, weights=None):
    """
    Plot 2D free energy function.

    Parameters
    ----------
    path_to_cv1 : str
        Path to the file containing first collective variable values.
    path_to_cv2 : str
        Path to the file containing second collective variable values.
    nbins : int
        Number of bins.
    cv1_label : str
       Label of the first collective variable.
    cv2_label : str
        Label of the second collective variable.
    output_name : str
        Name of the output figure.
    weights : str
       Path to the file containing weights.
    """

    # Check if the weights are provided
    if weights is not None:
        weights = np.loadtxt(weights)

    f, xedges, yedges = prepare_two_cvs(path_to_cv1, path_to_cv2, nbins, weights)
    plot_2D_contour(f, xedges, yedges, cv1_label, cv2_label, output_name)

    return
