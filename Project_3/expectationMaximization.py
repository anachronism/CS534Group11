import numpy as np 

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


testMean = [1, 10,100]
testCov = np.diag([0.1,0.1,0.1])
print 'full random: ',genMultidimGaussianData(2,2)
print 'Specified mean and cov: ', genMultidimGaussianData(3,5, mean=testMean, cov=testCov)