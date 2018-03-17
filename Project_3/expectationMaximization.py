import numpy as np
import argparse

class clusterCandidate:
	def __init__(self,mean,cov,logLike):
		self.mean = mean
		self.cov = cov
		self.logLike = logLike

## Produces list of tuples, each tuple being a data point.
def genMultidimGaussianData(nDims,nPoints,**keywordParameters):
	if 'mean' in keywordParameters:
		mean = keywordParameters['mean']
	else:
		mean = np.random.rand(nDims) * 5 ## This * 5 shows the range that data can be on.
	if 'cov' in keywordParameters:
		cov = keywordParameters['cov']
	else:
		cov = np.diag(np.random.rand(nDims)) 
	output = []
	for i in range(0,nPoints):
		output.append(tuple(np.random.multivariate_normal(mean,cov)))
	return output


parser = argparse.ArgumentParser(description='''CS 534 Assignment 3.''')
parser.add_argument('--n',dest='nClusters',nargs=1, type=int, default=3, help='''
										s	Number of clusters to find. input X to have the algorithm choose.
											''')

args = parser.parse_args()

numClusters = args.nClusters
numRestarts = 100 # Currently arbitrarily picked number
clusterOptions = []
dataDim = 2 ### TODO: make so this number is updated to the dimension of the input data.
if numClusters == 'X':
	## EM with Bayesian information criterion.
	pass # replace pass with actual thing.
else:
	## Standard EM ### TODO: determine if this is an either or, or if you just use BIC and then run EM.

	### EM Steps:	
	for i in range(0,numRestarts):
		pass # Replace pass with algorithm
		runEM = True
		# Randomly pick N means and covariances
		meanIn = np.random.rand(dataDim) ###TODO: MAKE SO IT SPANS RANGE OF INPUT VALUES
		covIn = np.diag(np.random.rand(dataDim))
		clusterOptions = clusterCandidate(meanIn,covIn,-float("inf"))

		while runEM == True:
			runEM = False # Make this conditional
			# Given data, assign data to clusters.
			# Given the points assigned to the clusters, update cluster mean and cov
			# Check log likelihood (save log likelihood)
			# If change in log likelihood is less than certain value, move to next iteration
			# or if count is too large.
	# Pick model with best log-likelihood


### OUTPUTS:
# Best fitting cluster centers
# Log-likelihood of the model