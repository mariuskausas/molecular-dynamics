#!/usr/bin/python3

import numpy as np
import mdtraj as mdt
import matplotlib.pyplot as plt


def load_txt(path_to_txt):
	""" Load .txt file."""
	return np.loadtxt(path_to_txt)


def get_significant_indices(path_to_weights, std):
	""" Extract indices of significant weights."""
	wopt = load_txt(path_to_weights)
	std_cutoff = wopt.mean() + std * wopt.std()
	significant_idx = np.where(wopt >= std_cutoff)[0]
	return wopt, significant_idx


def visualise_significant_weights(wopt, std):

	tick_params = dict(labelsize=22, length=10, width=1)

	fs = 20
	fig = plt.figure(figsize=[8, 5])
	ax = fig.add_subplot(111)

	plt.plot(wopt, label="Optimized weights", color="k", zorder=1)
	plt.hlines(wopt.mean(), xmin=0, xmax=wopt.shape[0], label="Mean", color="tab:blue", zorder=2, linewidth=2)
	plt.hlines(wopt.mean() + std * wopt.std(),
			xmin=0,
			xmax=wopt.shape[0],
			label= "{} $\\times$ $\\sigma$".format(std),
			color="tab:orange",
			zorder=2,
			linewidth=2)

	ax.tick_params(**tick_params)

	ax.legend(fontsize=tick_params["labelsize"] - 10, frameon=False)

	ax.set_xlabel("Indices", fontsize=fs)
	ax.set_ylabel("Optimised weights", fontsize=fs)

	plt.tight_layout()
	plt.savefig("significant_states_std_{}.png".format(std), dpi=300)


def save_significant_traj(path_to_traj, path_to_top, significant_idx):
	""" Save significant trajectory states."""
	traj = mdt.load(path_to_traj, top=path_to_top)
	significant_traj = traj[significant_idx]
	significant_traj.save_xtc("significant_traj.xtc")
	return
