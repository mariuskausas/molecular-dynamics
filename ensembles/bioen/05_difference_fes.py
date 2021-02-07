#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def get_histogram(xval, yval, nbins=100, weights=None):
	""" Compute 2D histogram of two given variables."""
	z, xedge, yedge = np.histogram2d(xval, yval, bins=nbins, weights=weights)
	x = 0.5 * (xedge[:-1] + xedge[1:])
	y = 0.5 * (yedge[:-1] + yedge[1:])
	return z.T, x, y


def density(z):
	""" Compute a probability density function."""
	return z / float(z.sum())


def free_energy(z):
	""" Log-transform of the probability density function."""
	prob = density(z)
	free_energy = np.inf * np.ones(shape=z.shape)
	nonzero = prob.nonzero()
	free_energy[nonzero] = -np.log(prob[nonzero])
	return free_energy


def convert_fes(cv1, cv2, bins, weights=None):
	""" Prepare input variables and convert to 2D histogram to FES."""
	H, xedges, yedges = get_histogram(cv1, cv2, bins, weights=weights)
	f = free_energy(H)
	return f, xedges, yedges


def load_txt(path_to_txt):
	""" Load .txt file."""
	return np.loadtxt(path_to_txt)


def difference_fes(path_to_cv1, path_to_cv2, path_to_weights, bins, cv1_label, cv2_label, output_name):
	""" Plot a difference FES."""
	# Load collective variables and weights
	cv1 = load_txt(path_to_cv1)
	cv2 = load_txt(path_to_cv2)
	weights = load_txt(path_to_weights)

	# Convert CVs to a matrix
	H, xedges, yedges = convert_fes(cv1, cv2, bins)
	Hw, xedgesw, yedgesw = convert_fes(cv1, cv2, bins, weights)

	# Normalize to zero
	H -= H.min()
	Hw -= Hw.min()

	# Remove infs
	Hinf = H[~np.isinf(H)]
	Hwinf = Hw[~np.isinf(Hw)]

	# Set contour level base
	levels_base = np.round(Hinf.max())

	# Set range for non-weighted and weighted matrices
	if Hinf.min() < Hwinf.min():
		vmin = np.floor(Hinf.min())
	else:
		vmin = np.floor(Hwinf.min())

	if Hinf.max() > Hwinf.max():
		vmax = np.ceil(Hinf.max())
	else:
		vmax = np.ceil(Hwinf.max())

	# Set font and figure dimensions
	fs = 20
	fig, (ax0, ax1, ax2) = plt.subplots(figsize=[22, 6], ncols=3)

	# Non-weighted
	im0 = ax0.contourf(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
					cmap="Spectral",
					levels=np.linspace(vmin, vmax, levels_base * 2 + 1),
					corner_mask=True,
					alpha=0.75)
	im0c = ax0.contour(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
					cmap="gist_gray",
					levels=np.linspace(vmin, vmax, levels_base * 2 + 1),
					linewidths=0.5,
					corner_mask=True,
					alpha=1)

	ax0.tick_params(labelsize=20)
	ax0.set_title("$-\ln p_{MD}$", fontsize=fs + 4)
	ax0.set_ylabel("{}".format(cv2_label), fontsize=fs)
	ax0.set_xlabel("{}".format(cv1_label), fontsize=fs)

	cbar0 = fig.colorbar(im0, ax=ax0)
	cbar0.ax.tick_params(labelsize=20)
	cbar0.set_ticks(np.arange(vmin, vmax + 1, 1))

	# Weighted
	im1 = ax1.contourf(Hw, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
					cmap="Spectral",
					levels=np.linspace(vmin, vmax, levels_base * 2 + 1),
					corner_mask=True,
					alpha=0.75)
	im1c = ax1.contour(Hw, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
					cmap="gist_gray",
					levels=np.linspace(vmin, vmax, levels_base * 2 + 1),
					linewidths=0.5,
					corner_mask=True,
					alpha=1)

	ax1.tick_params(labelsize=20)
	ax1.set_title("$-\ln p_{MaxEnt}$", fontsize=fs + 4)
	ax1.set_ylabel("{}".format(cv2_label), fontsize=fs)
	ax1.set_xlabel("{}".format(cv1_label), fontsize=fs)

	cbar1 = fig.colorbar(im1, ax=ax1)
	cbar1.ax.tick_params(labelsize=20)
	cbar1.set_ticks(np.arange(vmin, vmax + 1, 1))

	# Difference matrix
	diff = Hw - H

	# Remove nans
	diffnan = diff[~np.isnan(diff)]

	# Set range for difference matrix
	if np.abs(diffnan.min()) > np.abs(diffnan.max()):
		vmin_max_diff = np.ceil(np.abs(diffnan.min()))
	else:
		vmin_max_diff = np.ceil(np.abs(diffnan.max()))

	im2 = ax2.contourf(diff, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
					cmap="seismic_r",
					levels=np.linspace(-vmin_max_diff, vmin_max_diff, levels_base * 4 + 1),
					corner_mask=True,
					alpha=0.75)
	im2c = ax2.contour(diff, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
					cmap="gist_gray",
					levels=np.linspace(-vmin_max_diff, vmin_max_diff, levels_base * 4 + 1),
					linewidths=0.75,
					corner_mask=True,
					alpha=1)

	ax2.tick_params(labelsize=20)
	ax2.set_title("$\Delta G$ MaxEnt - MD", fontsize=fs + 4)
	ax2.set_ylabel("{}".format(cv2_label), fontsize=fs)
	ax2.set_xlabel("{}".format(cv1_label), fontsize=fs)

	cbar2 = fig.colorbar(im2, ax=ax2)
	cbar2.ax.tick_params(labelsize=20)
	cbar2.set_ticks(np.arange(-vmin_max_diff, vmin_max_diff + 1, vmin_max_diff / 2))

	plt.tight_layout()

	plt.savefig(output_name + ".png", dpi=300)
	plt.show()


path_to_cv1 = '/home/mariusk/Documents/papers.data/HOIP/manuscript.figures/making_figures/ensembles_RBR/2.bioen/HOIPwt/d1.txt'
path_to_cv2 = '/home/mariusk/Documents/papers.data/HOIP/manuscript.figures/making_figures/ensembles_RBR/2.bioen/HOIPwt/d2.txt'
path_to_weights = '/home/mariusk/Documents/papers.data/HOIP/manuscript.figures/making_figures/ensembles_RBR/2.bioen/HOIPwt/bioen_wopt_theta_100_HOIPwt_gromos54a8.txt'
bins=100
cv1_label = "D1"
cv2_label = "D2"
output_name = "fes_difference"

difference_fes(path_to_cv1, path_to_cv2, path_to_weights, bins, cv1_label, cv2_label, output_name)