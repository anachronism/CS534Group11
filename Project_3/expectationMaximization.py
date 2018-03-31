import numpy as np
import scipy.stats as sp
import argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import random
from copy import deepcopy

### PARAMETERS
THRESHREPEAT = 0.1 # Value that LL has to improve by to keep updating EM in specific iteration
NUMITERATIONS = 1e6 # Number of times to repeat EM before restarting again.
f_readDataFile = False
dataFile = 'sample EM data v9.csv' # relative path to data.

## Class containing one candidate set of means and covariances.
class clusterCandidate:
    def __init__(self,gaussInst,logLike,numDataPoints):
        
        self.normals = gaussInst # list of length M = numClusters
        self.probNormals = np.random.random(len(gaussInst)) 
        self.probNormals = self.probNormals / sum(self.probNormals)
        self.LL = logLike # 1 log likelihood value
        self.probTable = np.full((numDataPoints,len(gaussInst)),0,dtype=np.float64)
                        # Array of size MxN N = numDatapoints
        self.normProbTable = np.full((numDataPoints,len(gaussInst)),1/len(gaussInst),dtype=np.float64) 
                            # Array of size MxN N = numDatapoints
        
    def expectationUpdate(self, data):
        numDataPoints = data.shape[0]
        for n in range(0, numDataPoints):
            #print n
            for m in range(0,len(self.normals)):
                self.probTable[n,m] = self.normals[m].pdf(data[n])
            
            for m in range(0,len(self.normals)):
                probSum = sum(np.multiply(self.probNormals,self.probTable[n]))
                self.normProbTable[n,m] = self.probNormals[m] * self.probTable[n,m]/probSum
                #rint self.normProbTable[n]
                if math.isnan(self.normProbTable[n,m]):
                    print 'here'
                    return -1

        return 1 
                    
    # From probTable, assign point to cluster based on which has highest value:
    def updateLL(self):
        newLL = 0

        for ind in range(0,self.probTable.shape[0]):
            newLL += np.log(np.sum(np.multiply(self.probNormals,self.probTable[ind,:]))) 

        self.LL = newLL

    # Maximization class function that re-calculates the mean by getting the summation of the probabilies
    # that datapoints are within a certain mean and then dividing that summation with the summation of
    # probabilities in the mean. 
    def maximizationUpdate(self, dataPoints):
        # Rows of normProbTable represents number of data points.
        numDataPoints = self.normProbTable.shape[0]
        # Columns of normProbTable represents number of means or number of clusters.
        numMeans = self.normProbTable.shape[1]
        dataDim = len(dataPoints[0,:])

        self.probNormals = np.zeros(numMeans)
        for m in range(0,numMeans):
            for n in range(0,numDataPoints):
                self.probNormals[m] += self.normProbTable[n, m]
        self.probNormals /= numDataPoints


        # Initialize summations to zero vector or zero.
        for j in range(0,numMeans):
            # Calculate New mean:
            summationProb = 0
            summationCov = np.zeros((dataDim,dataDim))
            summationProbDatapoints = np.zeros((1,dataDim))

            for i in range(0,numDataPoints):
                expectedValue = self.normProbTable[i,j]
                ### TODO: double check that the individual points are grabbed correctly from the structure
                currentData = dataPoints[i,:]
                expectedData = expectedValue * currentData
                summationProbDatapoints += expectedData
                summationProb += expectedValue

            newMean = summationProbDatapoints / summationProb
            newMean = newMean[0]

            for i in range(0,numDataPoints):
                expectedValue = self.normProbTable[i,j]
                currentData = dataPoints[i,:]
                diffMean = currentData-newMean
                eltCov = expectedValue *np.transpose(diffMean)*diffMean
                summationCov += np.diag(eltCov)
            newCov = summationCov/summationProb

           # print newCov
            for elt in newCov:
                #print elt
                for elt2 in elt:
                    if math.isnan(elt2):
                        print 'will break'
                        #print summationProb
            if abs(newCov[0,0]) < 1e-20:
            	#print newCov
            	#print self.normProbTable
                return -1
            else:
                #print newCov
                self.normals[j]= sp.multivariate_normal(newMean,newCov) 
        return 1

                             
### Plotting functions:
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
    numClusters = len(pointArray)
    #colors = cm.rainbow(np.linspace(0, 1, numClusters))
    colors = cm.get_cmap('brg', numClusters)
    i = 0
    for elt in pointArray:
        plt.scatter(elt[:,0],elt[:,1],c=colors(i),s=35)
        i = i+1
    plt.show()

### read data file
def readDataFile(fileLoc):
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
    return np.asarray(data)

# def calcDistance(point1, point2):
#     dimension = len(point1)
#     sum = 0
#     for i in range(0,dimension):
#         sum += math.pow((point1[i]-point2[i]),2) 
#     return math.sqrt(sum)
    
## 
def calcBIC(candidate):
    numDataPoints = candidate.normProbTable.shape[0]
    numParameters = candidate.normProbTable.shape[1] * (len(candidate.normals[0].mean) + np.diag(candidate.normals[0].cov).size) ### TODO: Verify that this is actual number of parameters.
    BICVal = np.log(numDataPoints) * numParameters - 2*candidate.LL ## Numpy log is ln.
    return BICVal   

## Produces list of Datapoints
def genMultidimGaussianData(nDims,nPoints,**keywordParameters):
    
    if 'meanRange' in keywordParameters:
        meanRange = keywordParameters['meanRange']
    else:
        meanRange = [0,1]

    if 'covRange' in keywordParameters:
        covRange = keywordParameters['covRange']
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
    


    gaussianInstance = sp.multivariate_normal(mean=mean,cov=cov)
    output = np.zeros((nPoints,nDims))
    for i in range(0,nPoints):
        output[i,:] = gaussianInstance.rvs()
    return output,gaussianInstance


## Given number of clusters, run EM.
def expectationMaximization(nRestarts,nClusters,dataDim,meanRange,covRange,pointsIn,**keywordParameters):
    global THRESHREPEAT
    global NUMITERATIONS
    clusterOptions = []

    for i in range(0,nRestarts):
        print '     EM Restart Number ', i
        iterationCount = 0
        runEM = True
        # Randomly pick N means and covariances
        gaussInstances = []

        if 'externalInputData' in keywordParameters:
            externalInputData = keywordParameters['externalInputData']
            if externalInputData:
                means = np.zeros((nClusters,dataDim))
                for i in range(0,nClusters):
                    ## Dumb init choice
                    randx = random.uniform(np.amin(pointsIn[:,0]),np.amax(pointsIn[:,0]))
                    randy = random.uniform(np.amin(pointsIn[:,1]),np.amax(pointsIn[:,1]))
                    means[i,:] = [randx,randy]

                    # ## smart init choice (init with point in dataset)
                    # pointNum =random.randint(0,len(pointsIn[:,0])-1)
                    # means[i,:]=pointsIn[pointNum,:]
                
        # Initialize distribution estimates.
        for i in range(0,nClusters):
            if(externalInputData):
                _,gaussInst = genMultidimGaussianData(dataDim,1,mean=means[i],cov=covRange)
            else:
                _,gaussInst = genMultidimGaussianData(dataDim,1,meanRange=meanRange,covRange=covRange)
            gaussInstances.append(gaussInst)

       # Initialize clusterCandidate class instance.
        currentClusterCandidate = clusterCandidate(gaussInstances,-float("inf"),len(pointsIn))
       
        while runEM == True:
            lastLL = currentClusterCandidate.LL
            flag = currentClusterCandidate.expectationUpdate(pointsIn) 
            if flag == -1:
                runEM = False
                currentClusterCandidate.LL = -float('inf')
            else:
                flag2 = currentClusterCandidate.maximizationUpdate(pointsIn)
                if flag2 == -1:
                    runEM = False
                    currentClusterCandidate.LL = -float('inf')
                else:
                    currentClusterCandidate.updateLL() 

            if (currentClusterCandidate.LL - lastLL < THRESHREPEAT) or (iterationCount > NUMITERATIONS): 
                runEM = False 
                iterationCount = 0
            else:
                iterationCount += 1
            
        print "         LL: ", currentClusterCandidate.LL
        clusterOptions.append(currentClusterCandidate)

    # Pick model with best log-likelihood
    savedLL = []
    for elt in clusterOptions:
        savedLL.append(elt.LL)

    retIndex = np.argmax(savedLL)    

    return clusterOptions[retIndex]


### MAIN:
parser = argparse.ArgumentParser(description='''CS 534 Assignment 3.''')
parser.add_argument('--n',dest='nClusters',nargs=1, type=str, default='3', help='''
                                        s    Number of clusters to find. input X to have the algorithm choose.
                                            ''')
parser.add_argument('--nRestarts',dest='nRestarts',nargs=1, type=int, default=10, help='''
                                        s    Number of restarts for EM. Default is 3.
                                            ''')

args = parser.parse_args()
if type(args.nClusters) == list:
    if args.nClusters[0] == 'X':
        numClusters = args.nClusters[0]
    else:
        numClusters = int(args.nClusters[0])
else:
    numClusters = int(args.nClusters)

if type(args.nRestarts) == list:
    numRestarts = int(args.nRestarts[0])
else:   
    numRestarts = int(args.nRestarts)

if f_readDataFile:
    testData = readDataFile(dataFile)
    listTestData = [testData]
    #plot2DClusters(listTestData)  
    dataDim = len(testData[0,:])
    numDataPoints = len(testData[:,0])
    externInput=True
    dataCovRange = [5,60]
    dataMeanRange = []

    
    # plt.scatter(testData[:,0],testData[:,1])
    # plt.show()
else:
    covOfInputData = False
    numDataPoints = 50 
    dataMeanRange = [-100,100] 
    dataCovRange = [ 10, 900] 
    dataDim = 2 
    externInput = False
    numTestClusters = 2
    testData = None
    for elt in range(numTestClusters):
        testDatatmp,testCluster = genMultidimGaussianData(dataDim,numDataPoints,meanRange=dataMeanRange,covRange=dataCovRange)
        if elt == 0:
            testData = testDatatmp
        else:
            testData = np.concatenate((testData,testDatatmp))
    plot2DClusters([testData])    


### Run EM:

if numClusters == 'X':
    ## EM with Bayesian information criterion.
    currentBIC = 0
    lastBIC = float("inf")
    numClusters_tmp = 2
    endThresh = 0
    oldCandidate = None
    justStarted = True
    while( lastBIC - currentBIC > endThresh):  
        print 'EM with ',numClusters_tmp, 'clusters running.'      
        # Run EM with random restarts.
        # Using resulting log likelihood, calculate BIC
        if justStarted == False:
            lastBIC = deepcopy(currentBIC)
        else:
            justStarted = False

        newCandidate = expectationMaximization(numRestarts,numClusters_tmp,dataDim,dataMeanRange,dataCovRange,testData,externalInputData=externInput) 

        currentBIC = calcBIC(newCandidate)
        ## BIC = ln(numDataPoints)*numParametersEst - 2 * log-likelihood
        if (lastBIC - currentBIC <= endThresh):
            retCandidate = oldCandidate
            retBIC = lastBIC
            retNumClusters = numClusters_tmp - 1
        else:
            oldCandidate = newCandidate
            numClusters_tmp = numClusters_tmp + 1
    
    
    ## Plots:

    

    ### RETURN: num clusters, LL, BIC, cluster centers.
    print '______RESULTS______'
    print 'Num Clusters: ',retNumClusters
    print 'Log Likelihood: ',retCandidate.LL
    print 'BIC: ',retBIC
    
    for ind,elt in enumerate(retCandidate.normals):
        print 'Mean ',ind,': ',elt.mean
        print 'Cov ',ind,': \n',elt.cov,'\n'

    clusteredPoints = dividePoints(retCandidate.normProbTable,testData)
    plot2DClusters(clusteredPoints)
    


else:
    ## Standard EM 
    ### EM Steps:
    print 'EM with ',numClusters, 'clusters running.'
    retCandidate = expectationMaximization(numRestarts,numClusters,dataDim,dataMeanRange,dataCovRange,testData,externalInputData=True)   


    ### OUTPUTS:
    # Best fitting cluster centers
    print '______RESULTS______'
    # Log-likelihood of the model
    print 'Num Clusters: ', numClusters
    print 'Log Likelihood: ',retCandidate.LL

    for ind,elt in enumerate(retCandidate.normals):
        print 'Mean ',ind,': ',elt.mean
        print 'Cov ',ind,': \n',elt.cov,'\n'

    clusteredPoints = dividePoints(retCandidate.normProbTable,testData)
    plot2DClusters(clusteredPoints)
