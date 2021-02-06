d = """
=============================================

Calculate overlap between multiple essential subspaces using root mean squared inner product (RMSIP)

Reference: Cossio-Perez, R. et al (2017), J. Chem. Inf. Model, 57, 826-834.

The script provides a way to calculate similarity between multiple simulated trajectories based
on their essential subspace overlap. One can select a number of top eigenvectors for overlap calculation. 

Before use:
- Install mdtraj, numpy and sklearn

Marius Kausas					   2019 11 01
=============================================
"""


import mdtraj as mdt
import numpy as np
from sklearn.decomposition import PCA


def traj_prep(top, traj):

	"""
	Prepare input for PCA analysis.

	Trajectory is parsed in such a way that only CA atoms are extracted
	and all frames are superposed on initial frame. Finally, all XYZ CA atom coordinates
	are reshaped for correct PCA input.

	:param top: Topology file, as .pdb file
	:param traj: Trajectory file, as .xtc, .dcd or others
	:return: Numpy array of reshaped XYZ coordinates (number of frames, number of atoms * 3)
	"""

	# Load topology file
	pdb = mdt.load_pdb(top)

	# Extract CA atoms from a topology
	calphas = pdb.topology.select("name CA")
	ca_pdb = pdb.atom_slice(calphas)

	# Superpose trajectory trajectory
	traj = mdt.load(traj, top=top, atom_indices=calphas)
	traj.superpose(ca_pdb)

	# Reshape XYZ coordinates
	pca_input = np.reshape(traj.xyz, (traj.xyz.shape[0], traj.xyz.shape[1] * 3))

	return pca_input


def get_eigenvectors(data, n_components):

	"""
	Perform PCA analysis.

	:param data: Numpy array of reshaped XYZ coordinates
	:param n_components: Number of top PCA eigenvectors (int)
	:return: Numpy array of top eigenvectors
	"""

	# Perform PCA
	pca = PCA(n_components=n_components)
	pca.fit_transform(data)

	return pca.components_


def rmsip(top, traj1, traj2, n_components):

	"""
	Calculates RMSIP score for given two trajectories.

	:param top: Topology file, as .pdb file
	:param traj1: First trajectory file, as .xtc, .dcd or others
	:param traj2: Second trajectory file, as .xtc, .dcd or others
	:param n_components: Number of top PCA eigenvectors (int)
	:return: RMSIP score (float)
	"""

	# Prepare PCA inputs
	pca_input1 = traj_prep(top, traj1)
	pca_input2 = traj_prep(top, traj2)

	# Perform PCA and get eigenvectors
	eigenvectors1 = get_eigenvectors(pca_input1, n_components)
	eigenvectors2 = get_eigenvectors(pca_input2, n_components)

	# Calculate RMSIP
	inner_product = 0
	for i in range(n_components):
		for j in range(n_components):
			inner_product += np.power(np.dot(eigenvectors1[i], eigenvectors2[j]), 2)
	rmsip = np.sqrt(inner_product/float(n_components))

	return rmsip


def pairwise_rmsip(top, trajset1, trajset2, n_components):

	"""
	Calculate pairwise rmsip between two sets of trajectories.

	:param top: Topology file, as .pdb file
	:param trajset1: List of paths to trajectories as str
	:param trajset2: List of paths to trajectories as str
	:param n_components: Number of top PCA eigenvectors (int)
	:return: Pairwise RMSIP matrix
	"""

	# Define pairwise rmsip matrix
	size_trajset1 = len(trajset1)
	size_trajset2 = len(trajset2)
	pairwise_mat = np.zeros((size_trajset1, size_trajset2))

	# Calculate pairwise rmsip
	for i in range(size_trajset1):
		for j in range(size_trajset2):
			pairwise_mat[i:i + 1, j:j + 1] = rmsip(top, trajset1[i], trajset2[j], n_components)

	return pairwise_mat


def discretize_pairwise_rmsip(pairwise_mat, bins):

	"""
	Discretize pairwise RMSIP matrix for provided bins.

	:param pairwise_mat: Numpy array of a pairwise RMSIP matrix
	:param bins: List of bins. Range must be [0, 1]
	:return: Discretized RMSIP matrix
	"""

	# Generate empty discretized matrix
	discretized_mat = np.zeros((pairwise_mat.shape[0], pairwise_mat.shape[1]))

	# Discretize a pairwise RMSIP matrix
	bins = np.array(bins)
	for i in range(len(bins) - 1):

		# Define lower and upper bounds for bins
		lower = bins[i:i+2][0]
		upper = bins[i:i+2][1]

		# Populate newly generated matrix with discretized values
		discretized_mat[np.logical_and(pairwise_mat >= lower, pairwise_mat < upper)] = upper

	return discretized_mat


def reswise_loading(eigenvec_mat, eigenvec_num):

	"""
	Calculate residue contribution (loading) for a selected eigenvector.

	:param eigenvec_mat: Numpy array of igenvector matrix
	:param eigenvec_num: Eigen vector number (int)
	:return: Numpy array of residue loadings
	"""

	# Select an eigenvector
	selected_eigenvec = eigenvec_mat.T[:, eigenvec_num - 1:eigenvec_num]
	reshaped = selected_eigenvec.reshape(eigenvec_mat.shape[1] / eigenvec_mat.shape[0], eigenvec_mat.shape[0])

	# Calculate the norm of xyz eigenvector
	residue_loading = np.sqrt(np.sum(np.power(reshaped, 2), axis=1))

	return residue_loading
