#!/usr/bin/python3
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd


def experimental_data_prep(fit):
	""" Format experimental data."""
	np.savetxt(fname="./input.files/experimental_data/exp-saxs.dat", X=fit, fmt="%.6E")
	return


def read_crysol_fit(path_to_file, maxq):
	""" Read CRYSOL .fit output."""
	fit = pd.read_csv(path_to_file, delim_whitespace=True, skiprows=1, names=["q", "Iqexp", "sigmaexp", "fit"])
	fit = fit[fit["q"] <= maxq]
	return fit.values


def format_fits(path_to_fits, maxq, output_dir):
	""" Format theoretical fits for BioEn simulated data input."""
	fits = glob.glob(path_to_fits)
	fits.sort()
	exp_scattering = read_crysol_fit(fits[0], maxq)[9:, :3]
	experimental_data_prep(exp_scattering)
	q = exp_scattering[:, :1]
	for fit in fits:
		fit_idx = re.findall(r"\d+", os.path.basename(fit))[0][:-2]
		theoretical_scattering = read_crysol_fit(fit, maxq)[:, 3:]
		stacked_array = np.hstack((q, theoretical_scattering))
		print(output_dir + fit_idx + ".dat")
		np.savetxt(fname=output_dir + fit_idx + ".dat", X=stacked_array, fmt="%.6E")
	return


format_fits(path_to_fits="/fits/*.fit", maxq=0.20, output_dir="./input.files/simulated_data/")