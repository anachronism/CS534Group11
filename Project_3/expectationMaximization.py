import numpy as np
import argparse
import math
class clusterCandidate:
	def __init__(self,means,cov,logLike,numDataPoints):
		
		self.means = means # list of length M = numClusters
							# each is a tuple.
		self.cov = cov # matr of size NxN = covariance
		self.LL = logLike # 1 log likelihood value
		self.dataLabels = np.full((numDataPoints,len(means[0])),-1) ### TODO: Make reasonable.
							# Array of size MxN N = numDatapoints

#read data file
def readPriceTable(fileLoc):
    firstLineFlag = 1
    data = []           
            
    with open(fileLoc,'r') as f:
        for line in f:
            row = (line.strip('\n')).split(',')
            #print row
            if(firstLineFlag):
                size = len(row)          
                firstLineFlag = 0
            data.append((row))
            
            #for index,elt in enumerate(row):
    print data[0:10]        
    return data

def calcDistance(point1, point2):
	dimension = len(point1)
	sum = 0
	for i in range(0,dimension):
		sum += math.pow((point1[i]-point2[i]),2) 
	return math.sqrt(sum)

## 
def calcBIC(candidate):
    numDataPoints = candidate.dataLabels.shape[0]
    numParameters = candidate.dataLabels.shape[1] ### TODO: Verify that this is actual number of parameters.
    BICVal = np.log(numDataPoints) * numParameters - 2*candidate.LL ## Numpy log is ln.
    return BICVal   

## Produces list of tuples, each tuple being a data point.
def genMultidimGaussianData(nDims,nPoints,**keywordParameters):
	
	if 'meanRange' in keywordParameters:
		meanRange = keywordParameters['meanRange']
	else:
		meanRange = [0,1]

	if 'covRange' in keywordParameters:
		covRange = keywordParameters['meanRange']
	else:
		covRange = [0, 0.1]
	
	if 'mean' in keywordParameters:
		mean = keywordParameters['mean']
	else:
		mean = np.random.uniform(meanRange[0],meanRange[1],nDims) ## This * 5 shows the range that data can be on.
	
	if 'cov' in keywordParameters:
		cov = keywordParameters['cov']
	else:
		cov = np.diag(np.random.uniform(covRange[0],covRange[1],nDims)) 
	
	output = []
	for i in range(0,nPoints):
		output.append(tuple(np.random.multivariate_normal(mean,cov)))
	return output,mean,cov



### MAIN:


print calcDistance([2, 5, 1], [1,2,3])

print asdfasdf
parser = argparse.ArgumentParser(description='''CS 534 Assignment 3.''')
parser.add_argument('--n',dest='nClusters',nargs=1, type=int, default=3, help='''
										s	Number of clusters to find. input X to have the algorithm choose.
											''')

args = parser.parse_args()

numClusters = args.nClusters
numRestarts = 100 # Currently arbitrarily picked number
clusterOptions = []

numDataPoints = 50 # TODO: make this part of reading in the text file
dataDim = 2 ### TODO: make so this number is updated to the dimension of the input data.
dataMeanRange = [0,1] ### TODO: make this based on input data [min,max]
dataCovRange = [ 0, 0.1] ### TODO: make this based on input data



if numClusters == 'X':
	## EM with Bayesian information criterion.
    currentBIC = 0
    lastBIC = -1
    numClusters_tmp = 2
    endThresh = 0
    oldCandidate = None
    while (currentBIC - lastBIC > endThresh):        
	    # Run EM with random restarts.
	    # Using resulting log likelihood, calculate BIC
	    newCandidate = runEM(numRestarts,numClusters_tmp) ### TODO: Update with actual inputs that will be needed.
	    lastBIC = currentBIC
	    currentBIC = calcBIC(newCandidate)
		## BIC = ln(numDataPoints)*numParametersEst - 2 * log-likelihood
	    if (currentBIC - lastBIC <= endThresh):
	        retCandidate = oldCandidate
	        retBIC = lastBIC
	        retNumClusters = numClusters_tmp - 1
	    else:
	        oldCandidate = newCandidate
	        numClusters_tmp = numClusters_tmp + 1
	
	### RETURN: num clusters, LL, BIC, cluster centers.

else:
	## Standard EM 
    ### TODO: MOVE THE WHOLE THING (INCLUDING RESTARTS) INTO A FUNCTION SO BIC VERSION CAN CALL.
	### EM Steps:	
	for i in range(0,numRestarts):
		pass # Replace pass with algorithm
		runEM = True
		# Randomly pick N means and covariances
		meansIn = []
		covIn = []
		for i in range(0,numClusters):
			_,meansTmp,covTmp = genMultidimGaussianData(dataDim,1,meanRange=dataMeanRange,covRange=dataCovRange) ### TODO: set meanRange and covRange based on input data. 
			meansIn.append(meansTmp)
			covIn.append(covTmp)		
		
		clusterOptions.append(clusterCandidate(meansIn,covIn,-float("inf"),numDataPoints))
		while runEM == True:
			runEM = False ### TODO: \ Make this conditional
			# Given data, assign data to clusters.
			# Given the points assigned to the clusters, update cluster mean and cov
			# Check log likelihood (save log likelihood)
			# If change in log likelihood is less than certain value, move to next random restart
			# or if count is too large.
	# Pick model with best log-likelihood
	

### OUTPUTS:
# Best fitting cluster centers
# Log-likelihood of the model
