#!/usr/bin/python3

d = """
=============================================
Set-up of grid-search variational studies for SAXS-biased coarse-grained 
native structure-based model (SBM) molecular dynamics simulations (MD).

Reference: Marie Weiel, Ines Reinartz and Alexander Schug, 2019, PLoS Comput Biol 15(3): e1006900.

The script will create a set of directories containing modified .mdp
files for setting up grid-search variation studies in order to identify a
suitable force constant for performing SAXS-biased SBM simulations.

One has to provide a set of temperatures, a .mdp file, define upper and lower
limit of force constants and provide additional simulation files, 
such as initial .gro and target scattering profile .diff to complete the setup of SBM simulations.

Example usage:

./gs_setup.py -temp "70,90" -mdp "../input.mdp/run.mdp" -lower_fc "1e-18" -upper_fc "1e-13" -sim_files "../input.sim_files/" 

The above command will generate two folders t70 and t90, each containing 45 folders. Each folder 
will have a modified .mdp file with newly set ref_t and vel_gen temperatures and waxs-fc constants.
The waxs-fc constants are generated on a log scale. For a one log difference between force constants
you will get 10 force constant spaced out on log scale. For a two log difference you will get 19 force constants, 
whereas for three log difference, one gets 27 force constants and so on.

Before use:
- Install numpy.

Marius Kausas					   2019 05 17
=============================================
"""


import os
import shutil
import argparse
import numpy as np


def read_mdp(mdp):

	""" Read and return .mdp file line by line."""

	with open(mdp) as f:
		read_data = f.readlines()
	f.close()

	return read_data


def generate_fc(lower, upper):

	""" Return a set of force constants spaced on a log scale."""

	# Here np.round is used to avoid np.log10 rounding error,
	# when there is one log difference between upper and lower force constants.
	num = np.round(np.log10(upper / lower) * 9 + 1, 1)
	fcs = np.geomspace(lower, upper, num=num)

	return fcs


def copy_files(src, dst):

	""" Copy files from a directory to a given destination."""

	# Define source files
	src_files = os.listdir(src)
	# Copy each file to a destination
	for file_name in src_files:
		full_file_name = os.path.join(src, file_name)
		if os.path.isfile(full_file_name):
			shutil.copy(full_file_name, dst)

	return


def generate_grid_search(temps, mdp,  lower_fc, upper_fc, sim_files):

	""" Generate a set of modified .mdp files with a set of temperatures and force constants."""

	# Read .mdp file and generate a set of force constants
	initial_mdp = read_mdp(mdp)
	fcs = generate_fc(lower_fc, upper_fc)
	# Iterate for each temperature and force constant
	for temp in temps:
		# Define a destination for a single temperature
		temp_dst = "t" + str(temp)
		os.mkdir(temp_dst)
		for i in range(len(fcs)):
			# Define a destination of a single force constant
			fc_dst = os.path.join(temp_dst, "", str(i))
			os.mkdir(fc_dst)
			with open("run.mdp", "a")as f:
				new_mdp = initial_mdp.copy()
				# Change .mdp force constant
				new_mdp[9] = new_mdp[9][:26] + str(fcs[i]) + "\n"
				# Change .mdp temperatures for ref_t and vel_gen
				new_mdp[44] = new_mdp[44][:21] + str(temp) + "\n"
				new_mdp[51] = new_mdp[51][:26] + str(temp) + "\n"
				f.write("".join(new_mdp))
			# Move newly generated .mdp and simulation files to a single force constant destination
			shutil.move("run.mdp", fc_dst)
			copy_files(sim_files, fc_dst)

	return


if __name__ == "__main__":

	# Argument parser

	argparser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d)
	argparser.add_argument("-temp", type=str, help="A set of temperatures", required=True)
	argparser.add_argument("-mdp", type=str, help="Path to .mdp", required=True)
	argparser.add_argument("-lower_fc", type=str, help="Lower limit for a force constant.", required=True)
	argparser.add_argument("-upper_fc", type=str, help="Upper limit for a force constant.", required=True)
	argparser.add_argument("-sim_files", type=str, help="Simulation files to be copied.", required=True)

	# Parse arguments

	args = argparser.parse_args()
	temperatures = args.temp.split(",")
	mdp_file = args.mdp
	lower_force = float(args.lower_fc)
	upper_force = float(args.upper_fc)
	simulation_files = args.sim_files

	# Generate initial .mdp files for grid search

	generate_grid_search(temperatures, mdp_file, lower_force, upper_force, simulation_files)
