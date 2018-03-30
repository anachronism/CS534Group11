import numpy as np
import scipy.stats as sp
import argparse
import matplotlib.pyplot as plt
import math
import random
from copy import deepcopy

THRESHREPEAT = 0.1 # Value that LL has to improve by to keep updating EM in specific iteration
NUMITERATIONS = 1e6 # Number of times to repeat EM before restarting again.

class clusterCandidate:
    def __init__(self,gaussInst,logLike,numDataPoints):
        
        self.normals = gaussInst # list of length M = numClusters
        self.probNormals = np.random.random(len(gaussInst)) 
        self.probNormals = self.probNormals / sum(self.probNormals)
        self.LL = logLike # 1 log likelihood value
        self.probTable = np.full((numDataPoints,len(gaussInst)),-1,dtype=np.float64)
                        # Array of size MxN N = numDatapoints
        self.normProbTable = np.full((numDataPoints,len(gaussInst)),-1,dtype=np.float64) 
                            # Array of size MxN N = numDatapoints
        
    def getProbabilities(self, data):
        #print data
        #print self.probTable[0]
        
        for n in range(0, numDataPoints):
            for m in range(0,len(self.normals)):
                self.probTable[n,m] = self.normals[m].pdf(data[n])
            # print "__prob___"
            #print np.log10(self.probTable[m])
            #print self.probTable[n]
            #print probSum
            
            #print probSum
            for m in range(0,len(self.normals)):
                probSum = sum(np.multiply(self.probNormals,self.probTable[n]))
                self.normProbTable[n,m] = self.probNormals[m] * self.probTable[n,m]/probSum   
                # if self.normProbTable[n,m] < 1e-6:
                #     print 'Normtable:',self.normProbTable[n,m]
                #     print self.probTable[n]
        	#print "__norm___"
            #print self.normProbTable[m]
            
# From probTable, assign point to cluster based on which has highest value:
    def updateLL(self):
        ### TODO: Check if there's a multiply needed here.
        newLL = 0

       # print self.normProbTable.shape[0]
        for ind in range(0,self.probTable.shape[0]):
            newLL += np.log(np.sum(np.multiply(self.probNormals,self.probTable[ind,:]))) 
            #print 'table',self.normProbTable[:,ind]## TODO: Check if there should be a multiply here.
        
        print newLL
        self.LL = newLL

    # Maximization class function that re-calculates the mean by getting the summation of the probabilies
    # that datapoints are within a certain mean and then dividing that summation with the summation of
    # probabilities in the mean. 
    def maximization(self, dataPoints):
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

            # print 'newCov',j,':',newCov
            ### TODO: need to calculate new covariance
            self.normals[j]= sp.multivariate_normal(newMean,newCov) 
            # self.normals[j].cov = newCov # = sp.multivariate_normal(newMean, newCov)
                
                

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

def calcDistance(point1, point2):
    dimension = len(point1)
    sum = 0
    for i in range(0,dimension):
        sum += math.pow((point1[i]-point2[i]),2) 
    return math.sqrt(sum)


def sortPointsWithMeans(data,meanCenters):
    pointDistances = np.zeros((data.shape[0],len(meanCenters)))
    for i in range(0,data.shape[0]):
        for j in range(0,len(meanCenters)):
            pointDistances[i,j] = calcDistance(data[i,:],meanCenters[j])

    #print pointDistances[1:10,:]
    clusterAssign = pointDistances.argmin(axis=1)
    ret = []
    for i in range(0,len(meanCenters)):
        print max(clusterAssign == i)
        ret.append(data[clusterAssign == i])

    return ret
    

## 
def calcBIC(candidate):
    numDataPoints = candidate.normProbTable.shape[0]
    numParameters = candidate.normProbTable.shape[1] ### TODO: Verify that this is actual number of parameters.
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
        #mean = np.full( nDims,mean)
    else:
        #print meanRange[0]
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


def expectationMaximization(nRestarts,nClusters,dataDim,meanRange,covRange,pointsIn,**keywordParameters):
    global THRESHREPEAT
    global NUMITERATIONS
    clusterOptions = []

    if 'covOfInputData' in keywordParameters:
        covOfInputData = keywordParameters['covOfInputData']
    else:
        covOfInputData = False

    ### TODO: make covariance init localized.


    for i in range(0,nRestarts):
        iterationCount = 0
        runEM = True
        # Randomly pick N means and covariances
        gaussInstances = []

        if 'initMeans' in keywordParameters:
            means = np.zeros((nClusters,dataDim))
            for i in range(0,nClusters):
                # randx = random.uniform(np.amin(pointsIn[:,0]),np.amax(pointsIn[:,0]))
                # randy = random.uniform(np.amin(pointsIn[:,1]),np.amax(pointsIn[:,1]))
                # means[i,:] = [randx,randy]

                pointNum =random.randint(0,len(pointsIn[:,0])-1)
                means[i,:]=pointsIn[pointNum,:]



            #splitPoints = sortPointsWithMeans(pointsIn,means)
            covIn = []
            for i in range(0,nClusters):
                varIn = []
                #covIn.append(np.diag(np.var(splitPoints[i],0)))
                for j in range(dataDim):
                    varIn.append(random.uniform(5,60))
                #print varIn
                covIn.append(np.diag(varIn))
                #print covIn
           
           # print 'Cov: ', covIn

        for i in range(0,nClusters):
            if(covOfInputData):
               # covIn = np.diag(np.var(pointsIn,0))
                # print covIn[i]
                _,gaussInst = genMultidimGaussianData(dataDim,1,mean=means[i],cov=covIn[i])
            else:
                _,gaussInst = genMultidimGaussianData(dataDim,1,meanRange=meanRange,covRange=covRange)
            gaussInstances.append(gaussInst)


        cntRuns = 0
        currentClusterCandidate = clusterCandidate(gaussInstances,-float("inf"),numDataPoints)
        #currentClusterCandidate.updateLL()
        while runEM == True:
            currentClusterCandidate.getProbabilities(pointsIn) # Given data, assign data to clusters.
            # Given the points assigned to the clusters, update cluster mean and cov
            currentClusterCandidate.maximization(pointsIn)
            # Check log likelihood (save log likelihood)
            lastLL = currentClusterCandidate.LL
            currentClusterCandidate.updateLL() 

            if (currentClusterCandidate.LL - lastLL < THRESHREPEAT) or (iterationCount > NUMITERATIONS) or (currentClusterCandidate.LL == -float("inf")): 
                if currentClusterCandidate.LL - lastLL < 0:
                    print 'SHOULDN\'T HAPPEN: ', currentClusterCandidate.LL,lastLL
                    print currentClusterCandidate.probNormals
                runEM = False 
                iterationCount = 0
            else:
                iterationCount += 1
            
            # If change in log likelihood is less than certain value, move to next random restart
            # or if count is too large.

        clusterOptions.append(currentClusterCandidate)

    # Pick model with best log-likelihood
    savedLL = []
    for elt in clusterOptions:
        savedLL.append(elt.LL)
    print 'MAX: ', max(savedLL)
    # print savedLL
    retIndex = savedLL.index(max(savedLL))    

    for elt in clusterOptions[retIndex].normals:
        print elt.mean
        print elt.cov

    return clusterOptions[retIndex]


### MAIN:
parser = argparse.ArgumentParser(description='''CS 534 Assignment 3.''')
parser.add_argument('--n',dest='nClusters',nargs=1, type=str, default='3', help='''
                                        s    Number of clusters to find. input X to have the algorithm choose.
                                            ''')

args = parser.parse_args()
if type(args.nClusters) == list:
    if args.nClusters[0] == 'X':
        numClusters = args.nClusters[0]
    else:
        numClusters = int(args.nClusters[0])
else:
    numClusters = int(args.nClusters)

numRestarts = 100 # Currently arbitrarily picked number
f_readDataFile = True
dataFile = 'sample EM data v2.csv' # relative path to data.

    

if f_readDataFile:
    testData = readDataFile(dataFile)
    listTestData = [testData]
    #plot2DClusters(listTestData)  
    dataDim = len(testData[0,:])
    numDataPoints = len(testData[:,0])
    covOfInputData = True
    dataCovRange = [1,np.amax(testData)-np.amin(testData)] 
    dataMeanRange = [np.mean(testData) -np.mean(testData)/2,np.mean(testData)+np.mean(testData)/2]

    
    # plt.scatter(testData[:,0],testData[:,1])
    # plt.show()
else:
    covOfInputData = False
    numDataPoints = 50 
    dataMeanRange = [0,1] 
    dataCovRange = [ 0, 0.1] 
    dataDim = 2 
    testData,testCluster = genMultidimGaussianData(dataDim,numDataPoints,mean=[-2,2],cov=[[1,0],[0,1]])


### Run EM:

if numClusters == 'X':
    ## EM with Bayesian information criterion.
    currentBIC = 0
    lastBIC = -1
    numClusters_tmp = 2
    endThresh = 0
    oldCandidate = None
    while(currentBIC - lastBIC > endThresh):        
        # Run EM with random restarts.
        # Using resulting log likelihood, calculate BIC
        newCandidate = expectationMaximization(numRestarts,numClusters_tmp,dataDim,dataMeanRange,dataCovRange,testData,covOfInputData=covOfInputData,initMeans=True) 
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
    # retNumClusters
    LL_best = retCandidate.LL
    print LL_best
    # retBIC
    clusterCenters = []
    for elt in retCandidate.normals:
        clusterCenters.append(elt.mean)


else:
    ## Standard EM 
    ### EM Steps:
    bestClusterCandidate = expectationMaximization(numRestarts,numClusters,dataDim,dataMeanRange,dataCovRange,testData,covOfInputData=covOfInputData,initMeans=True)   
   # print bestClusterCandidate.probTable 
    clusteredPoints = dividePoints(bestClusterCandidate.normProbTable,testData)
    print 'Best LL: ',bestClusterCandidate.LL 
    plot2DClusters(clusteredPoints)   

    ### OUTPUTS:
    # Best fitting cluster centers
    clusterCenters = []
    for elt in bestClusterCandidate.normals:    
        clusterCenters.append(elt.mean) 
    # Log-likelihood of the model
    LL_best = bestClusterCandidate.LL
