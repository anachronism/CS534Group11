import numpy as np
import scipy.stats as sp
import argparse
import matplotlib.pyplot as plt
import math

class clusterCandidate:
	def __init__(self,gaussInst,logLike,numDataPoints):
		
		self.normals = gaussInst # list of length M = numClusters
							# each is a tuple.
		self.LL = logLike # 1 log likelihood value
		self.probTable = np.full((numDataPoints,len(gaussInst)),-1) ### TODO: Make reasonable.
							# Array of size MxN N = numDatapoints
		def getProbabilities(self, data):
			for n in (0, numDataPoints):
				for m in len(gaussInst):
					self.probTable[m,n] = normals[m].pdf(data[n])
				probSum = sum(self.probTable[:,n])
				for m in len(gaussInst):
					self.probTable[m,n] = self.probTable[m,n]/probSum


# From probTable, assign point to cluster based on which has highest value:
# returns M lists containing points that belong to the cluster.
def dividePoints(pTable,points):
	indResult = []
	pointsInCluster = []
	for i in range(0,pTable.shape[0]):
		valsSearch=pTable[i,:]
		indResult.append(np.argmax(valsSearch))
	for i in range(0,pTable.shape[1]):
		indInCluster = [j for j, x in enumerate(indResult) if x == i]
		pointsInCluster.append(points[indInCluster])
	return pointsInCluster

def plot2DClusters(pointArray):
	for elt in pointArray:
		plt.scatter(elt[:,0],elt[:,1])
	plt.show()

#read data file
def readPriceTable(fileLoc):
    firstLineFlag = 1
    data = []           
    floatRow = [] 
    with open(fileLoc,'r') as f:
        for line in f:
            floatRow = []
            row = (line.strip('\n')).split(',')

            for stringIndex in range(0,len(row)):
                floatRow.append(float(row[stringIndex]))
            
            data.append(deepcopy(floatRow))
            
            #for index,elt in enumerate(row):
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
		#mean = np.full( nDims,mean)
	else:
		mean = np.random.uniform(meanRange[0],meanRange[1],nDims) ## This * 5 shows the range that data can be on.
	
	if 'cov' in keywordParameters:
		cov = keywordParameters['cov']
	else:
		cov = np.diag(np.random.uniform(covRange[0],covRange[1],nDims)) 
	


	gaussianInstance = sp.multivariate_normal(mean=mean,cov=cov)
	output = np.zeros((nPoints,nDims))
	for i in range(0,nPoints):
		output[i,:] = gaussianInstance.rvs()
	return output,gaussianInstance


def expectationMaximization(nRestarts,nClusters,dataDim,meanRange,covRange,pointsIn):
	for i in range(0,nRestarts):
		pass # Replace pass with algorithm
		runEM = True
		# Randomly pick N means and covariances
		gaussInstances = []
		for i in range(0,nClusters):
			_,gaussInst = genMultidimGaussianData(dataDim,1,meanRange=meanRange,covRange=covRange) ### TODO: set meanRange and covRange based on input data. 
			gaussInstances.append(gaussInst)
		
		clusterOptions.append(clusterCandidate(gaussInstances,-float("inf"),numDataPoints))
		while runEM == True:
			runEM = False ### TODO: \ Make this conditional
			# Given data, assign data to clusters.
			# Given the points assigned to the clusters, update cluster mean and cov
			# Check log likelihood (save log likelihood)
			# If change in log likelihood is less than certain value, move to next random restart
			# or if count is too large.
	# Pick model with best log-likelihood
	clusteredPoints = dividePoints(clusterOptions[0].probTable,pointsIn)
	plot2DClusters(clusteredPoints)		


### MAIN:


print calcDistance([2, 5, 1], [1,2,3])

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
### TODO: Sub with actual data
testData,testCluster = genMultidimGaussianData(dataDim,numDataPoints,mean=[-2,2],cov=[[1,0],[0,1]])

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
	    newCandidate = expectationMaximization(numRestarts,numClusters_tmp,dataDim,dataMeanRange,dataCovRange,testData) ### TODO: Update with actual inputs that will be needed.
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
	bestClusterCandidate = expectationMaximization(numRestarts,numClusters,dataDim,dataMeanRange,dataCovRange,testData)	
		

### OUTPUTS:
# Best fitting cluster centers
# Log-likelihood of the model


## Test genMultidimGaussianData
# nTestPoints = 1000
# ptsTest,meansTest,covTest = genMultidimGaussianData(2,nTestPoints,mean=[5,4])
# ptsTest2,_,_= genMultidimGaussianData(2,nTestPoints,mean=[-3,1])
# xTest,yTest = zip(*ptsTest)
# xTest2,yTest2 =zip(*ptsTest2)
# # print xTest2,yTest2
# plt.axis([-6,6,-6,6])
# plt.scatter([xTest, xTest2],[yTest,yTest2])
# plt.show()
# #print ptsTest
# nPointsTest = 100
# testOut,_ = genMultidimGaussianData(2,nPointsTest,mean=[2,2])
# testOut2,_ = genMultidimGaussianData(2,nPointsTest,mean=[-2,-2])
# plt.scatter(testOut[:,0],testOut[:,1])
# plt.scatter(testOut2[:,0],testOut2[:,1])

# plt.show()

