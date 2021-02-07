#!/usr/bin/python2

import re
import glob 
import numpy as np
import pickle


# Path to experimental data
path_experimental_data = "./input.files/experimental_data"

# Path to simulated data
path_simulated_data = "./input.files/simulated_data"

# Define path to the output of the preparation
path_output_preparation = "./input.files/output_preparation"

# Define path to the output of BioEn
path_output_bioen = "./input.files/output_bioen"

# Define path to the output of the analysis
path_output_analysis = "./input.files/output_analysis"

# Define experimental q, I(q) and errors
fn_exp = "{}/exp-saxs.dat".format(path_experimental_data)
exp_1 = np.loadtxt(fn_exp)
exp_tmp = exp_1[:,0:2] # q and I(q)
exp_err_tmp = exp_1[:,2] # errors

# Define number of restraints
nrestraints = len(exp_1)


# Extract theoretical scattering profiles from each .dat file
def read_dat(file):
    """ Read .dat file."""
    fit = np.loadtxt(file)
    return fit


def get_fit_scattering(fit_file):
    """ Extract theoretical scattering from a .dat file."""
    return read_dat(fit_file)[:, 1:]


sim_tmp = dict()
fits = glob.glob(path_simulated_data + "/*")
fits.sort()

# Define number of restraints
for fit in fits:
    sim_tmp[int(re.findall(r"\d+", fit)[0])] = get_fit_scattering(fit).reshape(nrestraints,)

# Define theta series, for which to perform reweighting
theta_series = list(np.round([10e5, 10e4] + list(np.geomspace(10000, 10, 10)) + [10e-1, 0]))
np.savetxt('theta.dat', np.array(theta_series))

# Define number of ensemble members
model_ids = range(0, len(fits))
number_of_ensemble_members = len(model_ids)
np.savetxt('{}/models_scattering.dat'.format(path_output_preparation), model_ids)

# Let BioEn perform an initial optimization of the coefficient
coeff = 'initial-optimization'

# Write pickle file as input for BioEn
fn_out_pkl = '{}/input-bioen-scattering.pkl'.format(path_output_preparation)
with open(fn_out_pkl, 'wb') as fp:
    pickle.dump([coeff, nrestraints, exp_tmp, exp_err_tmp, sim_tmp], fp)
