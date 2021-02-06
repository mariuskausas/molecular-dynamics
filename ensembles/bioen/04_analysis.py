#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.set_context("paper")

# path to experimental data
path_experimental_data = "input.files/experimental_data"

# path to simulated data
path_simulated_data = "input.files/simulated_data"

# define path to the output of the preparation
path_output_preparation = "input.files/output_preparation"

# define path to the output of BioEn
path_output_bioen = "input.files/output_bioen"

# define path to the output of the analysis
path_output_analysis = "input.files/output_analysis"

# output of BioEn is a pkl file (contains all the necessary information)
bioen_pkl = "{}/bioen-scattering.pkl".format(path_output_bioen)


def read_bioen_pkl(bioen_pkl):
	""" Read BioEn pickle output file."""
	df = pd.read_pickle(bioen_pkl)
	return df.to_dict()


# Read pkl file
bioen_data = read_bioen_pkl(bioen_pkl)

# define theta series
theta_series = list(bioen_data.keys())
theta_series.sort()


def extract_scattering(bioen_data):
	""" Extract unweighted and weighted scattering profiles."""
	for theta in theta_series:
		exp = bioen_data[theta]["exp"]["scattering"]
		exp_err = bioen_data[theta]["exp_err"]["scattering"].reshape(exp.shape[0], 1)
		sim_wopt = bioen_data[theta]["sim_wopt"]["scattering"]
		stacked = np.hstack((exp, exp_err, sim_wopt))
		np.savetxt("bioen_scattering_wopt_theta_{}.dat".format(int(theta)), stacked)
	init = bioen_data[0]["sim_init"]["scattering"]
	stacked = np.hstack((exp, exp_err, init))
	np.savetxt("bioen_scattering_init.dat".format(int(theta)), stacked)
	return


def extract_chi2_skl(bioen_data):
	""" Extract theta, reduced chi-squared and relative entropy values."""
	with open("chi2_skl.txt", "w") as f:
		for theta in theta_series:
			chi2 = bioen_data[theta]['chi2'] / bioen_data[theta]['nrestraints']
			skl = - bioen_data[theta]['S']
			f.write("{}\t{}\t{}\n".format(int(theta), chi2 * 2, skl))
	return


def extract_wopt_cumsum(bioen_data):
	""" For each theta, extract cumulative sums of optimised weights."""
	for theta in theta_series:
		wopt = np.array(bioen_data[theta]["wopt"])
		wopt_sorted = np.sort(wopt, axis=0)[::-1]
		wopt_sorted_cumsum = np.cumsum(wopt_sorted)
		np.savetxt("bioen_wopt_cumsum_theta_{}.txt".format(int(theta)), wopt_sorted_cumsum)
	return	


def visualize_chi2_skl(bioen_data):
	""" Plot reduced reduced chi-squared values against relative entropy as a function of theta."""
	fs = 30
	fig = plt.figure(figsize=[12, 4])
	ax = fig.add_subplot(111)

	for theta in theta_series:
		chi2 = bioen_data[theta]['chi2'] / bioen_data[theta]['nrestraints']
		skl = - bioen_data[theta]['S']
		ax.scatter(skl, chi2, marker='o', s=80, label=theta)

	ax.set_xlabel(r'$S_{\mathrm{KL}}$', fontsize=fs)
	ax.set_ylabel(r'$\chi^{2}$', fontsize=fs)

	ax.tick_params(labelsize=30)

	plt.grid()

	ax.legend(ncol=2, fontsize=15, loc='center left', bbox_to_anchor=(1, 0.5))

	plt.tight_layout()
	plt.savefig("{}/bioen_chi2_skl.png".format(path_output_analysis), dpi=400)

	return


def chi2red(exp, theor, error):
	""" Calculate reduced chi square value between two scattering curves."""
	chi_value = np.sum(np.power((exp - theor) / error, 2)) / (exp.size - 1)
	return np.sum(chi_value)


def plot_residuals(q, exp, fit1, fit2, sigma, fitname1, fitname2, output):
	""" Plot scattering fit, residuals and a reduced chi-squared value."""
	# Calculate a reduced chi square value
	chi_value = str(np.around(chi2red(exp, fit2, sigma), decimals=2))

	# Plot a fit between experimental and theoretical scattering data
	ax1 = plt.subplot2grid((4, 3), (0, 0), colspan=3, rowspan=3)
	ax1.plot(q, np.log10(exp), label="Exp", color="k", linewidth=1)
	ax1.fill_between(q, np.log10(exp - sigma), np.log10(exp + sigma), color='black', alpha=0.2, label='Error',
					linewidth=2.0)
	ax1.plot(q, np.log10(fit1), label=fitname1, color="tab:red", linewidth=2.0)
	ax1.plot(q, np.log10(fit2), label=fitname2, color="tab:green", linewidth=2.0)
	ax1.tick_params(labelsize=6)
	ax1.set_xticklabels([])
	ax1.set_ylabel('$log_{10}(I_{q})$')
	ax1.legend(ncol=2, loc="upper right", fontsize=6)
	ax1.text(0.75, 0.75, s=('Optimized $\chi^{2}_{red}=$' + chi_value),
			horizontalalignment='center',
			verticalalignment='center',
			transform=ax1.transAxes, fontsize=15)
	plt.grid()

	# Plot fit residuals
	residuals1 = (exp - fit1) / sigma
	residuals2 = (exp - fit2) / sigma
	ax2 = plt.subplot2grid((4, 3), (3, 0), colspan=3)
	ax2.axhline(y=0, xmin=0, xmax=1, ls='--', color="k", zorder=2, linewidth=1)
	ax2.scatter(q, residuals1, s=2, color="tab:red", zorder=3)
	ax2.scatter(q, residuals2, s=2, color="tab:green", zorder=3)
	ax2.tick_params(labelsize=6)
	ax2.set_xlabel('$q$')
	ax2.set_ylabel('$(I\Delta)/\sigma$')
	ax2.set_title("Residuals", fontsize=8)

	plt.grid()

	plt.subplots_adjust(hspace=0.75, wspace=0.75)
	plt.tight_layout()

	plt.savefig("single_theta_{}_scattering.png".format(output), dpi=600)
	plt.close()
	return


def read_thetas(path_to_theta):
	""" Read thetas.dat file."""
	return np.loadtxt(path_to_theta)


def plot_single_theta_scattering(bioen_data, theta, output_name):
	""" Plot a scattering profile for a single theta from BioEN data."""

	bioen_theta = bioen_data[theta]
	q = bioen_theta["exp"]["scattering"][:, 0]
	exp = bioen_theta["exp"]["scattering"][:, 1]
	init = np.array(bioen_theta["sim_init"]["scattering"]).squeeze()
	wopt = np.array(bioen_theta["sim_wopt"]["scattering"]).squeeze()
	sigma = bioen_theta["exp_err"]["scattering"]
	plot_residuals(q=q, exp=exp, fit1=init, fit2=wopt, sigma=sigma,
				fitname2="$\\theta$={}".format(theta), fitname1="Init",
				output=str(theta) + "_" + output_name)
	return


def plot_all_thetas_scattering(bioen_data, theta_series, output_name):
	""" Plot single scattering profiles for all thetas from BioEN data."""
	for theta in theta_series:
		plot_single_theta_scattering(bioen_data, int(theta), output_name)
	return


def visualize_scattering_all_thetas(bioen_data):
	""" Plot experimental and ensemble optimised scattering profiles for all thetas."""
	fs = 22
	fig = plt.figure(figsize=[8, 4])
	ax = fig.add_subplot(111)

	theta_max = np.max(list(bioen_data.keys()))

	exp = bioen_data[theta_max]['exp']['scattering']
	ax.plot(exp[:, 0], np.log10(exp[:, 1]), color='black', linewidth=1.5, label='Exp.', zorder=2)

	a = np.linspace(0.1, 1.0, num=len(theta_series))
	for i, theta in enumerate(theta_series):
		sim = bioen_data[theta]['sim_wopt']['scattering']
		ax.plot(exp[:, 0], np.log10(sim), color='red', alpha=a[i], linewidth=2.0,
				label=theta, zorder=3)

	ax.set_xlabel(r'q [${\AA}^{-1}$]', fontsize=fs + 2)
	ax.set_ylabel(r'I(q)', fontsize=fs)

	ax.legend(ncol=2, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))

	plt.grid()

	plt.tight_layout()
	plt.savefig("{}/bioen_theta_series.png".format(path_output_analysis), dpi=400)

	return


def visualize_all_cum_dist(thetas, bioen_data):
	""" Plot cumulative weight sum distributions for all thetas."""
	fs = 22
	fig = plt.figure(figsize=[8, 4])
	ax = fig.add_subplot(111)

	for theta in thetas:

		nmodels = bioen_data[theta]["nmodels"]
		a = np.vstack((bioen_data[theta]["nmodels_list"], np.array(bioen_data[theta]["wopt"]).reshape(1, -1)))

		models = []
		for i, id in enumerate(np.argsort(a[1, :])[::-1]):
			if i == 0:
				models.append([a[:, id][0], float(a[:, id][1]), float(a[:, id][1])])
			else:
				models.append([a[:, id][0], float(a[:, id][1]), models[-1][2] + float(a[:, id][1])])

		models = np.array(models)
		ax.plot(range(1, nmodels + 1), models[:, 2], zorder=1, label=theta)

	ax.set_xticks([0.0, 10.0, 100.0, 1000.0, 10000.0])
	ax.set_xticklabels([r'$10^{0}$', r'$10^{1}$', r'$10^{2}$', r'$10^{3}$', r'$10^{4}$'], fontsize=fs)
	ax.set_xlabel("Number of configurations", fontsize=fs)
	ax.semilogx()
	ax.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=fs)
	ax.set_ylim(0, 1)
	ax.set_ylabel("Cumulative distribution", fontsize=fs)
	ax.legend(ncol=2, fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
	ax.grid()

	plt.tight_layout()
	plt.savefig('{}/bioen_all_cum_weights.png'.format(path_output_analysis), dpi=600, bbox_inches='tight')

	return


def save_weight_as_txt(bioen_data, basename, theta, weights):
	""" Save optimised weights as a .txt files."""
	np.savetxt("bioen_{}_theta_{}_".format(weights, int(theta)) + basename + ".txt", bioen_data[theta][weights])
	return


def extract_weights(bioen_data, output_name):
	""" Extract initial and optimised weights from a parsed BioEn pickle output file."""
	thetas = list(bioen_data.keys())
	thetas.sort()
	save_weight_as_txt(bioen_data, output_name, thetas[0], "w0")
	for theta in thetas:
		save_weight_as_txt(bioen_data, output_name, theta, "wopt")
	return


extract_scattering(bioen_data)
extract_chi2_skl(bioen_data)
extract_wopt_cumsum(bioen_data)
visualize_chi2_skl(bioen_data)
visualize_scattering_all_thetas(bioen_data)
visualize_all_cum_dist(theta_series=theta_series, bioen_data=bioen_data)
extract_weights(bioen_data=bioen_data, output_name="ensemble")
plot_all_thetas_scattering(bioen_data=bioen_data, theta_series=theta_series, output_name="ensemble")