import numpy
from part2 import * # This import is only until we actually incorporate together.
import random,time,copy,math

IND = 100
COM = 200
RES = 300
TOXIC = 400
SCENIC = 500


# randomCrossover()

class GeneticChild:

     def __init__(self, mapIn):
		self.map =  copy.deepcopy(mapIn)
		self.map,self.buildCost,self.locations = populateSiteMap(self.map)
		self.utilVal = calculateStateScore(self.map) - self.buildCost 





'''
Start: k list of random, valid states.
calculate fitness scores for these k states.
pick two states.
execute 
'''



def geneticStateSearch(originalMap,iCount,cCount,rCount,timeToRun):
	k = 100
	k2 = int(k/20)	
	numCull = 5 ### Or maybe make it so that it's a threshold
	prob_mutate = 0.06

	numRows = originalMap.shape[0]
	numCols = originalMap.shape[1]

	firstRun = True
	timeRun = 0.0
	initTime = time.time()
	lastGen = []
	currentGen =[]
	lastScores = []

	while timeRun < timeToRun:
		## First Population: Generate random states and save. 		
		if firstRun == True:
			# Generate  k states randomly	
			for i in range(0,k):
				lastGen.append(GeneticChild(originalMap))
				lastScores.append(lastGen[i].utilVal)
			firstRun = False
			print (zip(range(0,k),lastScores))
		else:
			currentGen = []
			# Select k2 most fit states to save
			zippedScores = zip(range(0,k),lastScores) 
			zippedScores.sort(key=lambda x: x[1])
			print 'SAVE'
			for i in range(1,k2+1):
				ind_elite = (zippedScores[k-i])[0]
				print lastGen[ind_elite].utilVal
				currentGen.append(lastGen[ind_elite])
			## 
			# cull from last gen
			print 'POP'
			for i in range(0,numCull):
				print((zippedScores[i])[1])
				lastGen.pop((zippedScores[i])[0])

			### TODO: ACTUALLY IMPLEMENT CROSSOVER
			# Use crossover etc to make k-k2 states
				# weight towards states w/ better fitness.
				# Combining 2 states makes 2 successors.
				# Randomly change some bits in some states.

			pass
			

		# Update the current time
		timeRun = time.time() - initTime


'''
Part 2 genetic testing
'''
random.seed()
originalMap,iCount,cCount,rCount = readFile('sample2.txt')
# test = GeneticChild(originalMap)
geneticStateSearch(originalMap,iCount,cCount,rCount, 1)