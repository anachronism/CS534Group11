import numpy
from collections import namedtuple
from part2 import * # This import is only until we actually incorporate together.
import random,time,copy,math

IND = 100
COM = 200
RES = 300
TOXIC = 400
SCENIC = 500

DEBUG_GENETICS = 1
# Named Tuple containing genetic search parameters.
GeneticParams = namedtuple("GeneticParams", "iCount cCount rCount pMutate pCross timeToRun k k2 numCull")

## Class containing one map candidate.
class GeneticChild:
	def __init__(self, mapIn,buildCost,locations,timeFound):
		self.map = mapIn#copy.deepcopy(mapIn)
		self.buildCost = buildCost
		self.locations = locations
		self.timeFound = timeFound
		self.utilVal = calculateStateScore(self.map) - self.buildCost

	## Mutliple inits, depending on if a list of locations is provided or not.
	@classmethod
	def fromRandom(cls, mapIn, timeFound):
     	# Can maybe skip this first step, and just feed mapIn into it.
		cls.map =  copy.deepcopy(mapIn)
		cls.timeFound = copy.deepcopy(timeFound)
		cls.map,cls.buildCost,cls.locations = populateSiteMap(cls.map)
		return cls(cls.map,cls.buildCost,cls.locations,cls.timeFound)

	@classmethod
	def fromLocations(cls, mapIn, locations,timeFound):
		cls.map = copy.deepcopy(mapIn)
		cls.locations = locations
		cls.timeFound = copy.deepcopy(timeFound)
		cls.map, cls.buildCost = changeSiteMap(cls.map,cls.locations)
		# print cls.locations
		# print cls.map
		return cls(cls.map,cls.buildCost,cls.locations,cls.timeFound)


## UTILITY FUNCTIONS
def checkValidLocation(potentialLoc,mapIn,listInvalid):
	if potentialLoc in listInvalid:
		return False
	elif mapIn[potentialLoc[0],potentialLoc[1]] == TOXIC:
		return False
	else:
		return True

def generateRandomLocation(mapIn,listInvalid):
	f_validLocation = False
	while not f_validLocation:
		randLoc = [random.randint(0,mapIn.shape[0]-1),random.randint(0,mapIn.shape[1]-1)]
		f_validLocation = checkValidLocation(randLoc,mapIn,listInvalid)
	return randLoc


def runCrossover(parent1, parent2, mapIn,params,initTime):
	# Combine the two randomly.
	locations_1 = []
	locations_2 = []

	for i in range(0,len(parent1.locations)):
		## Check if the location will mutate for either child, and picking parents.  
		randMutate1 = random.random()
		randMutate2 = random.random()
		randParent = random.random()

		## Evaluate the location for the first child.
		# If it should mutate, then add a random location.
		if randMutate1 > 1 - params.pMutate:
			locations_1.append(generateRandomLocation(mapIn,locations_1))
		# Else, if parent 1 is randomly chosen, make sure that the location it has is 
		# Unoccupied, and then add it. Otherwise, just mutate.
		else:	
			if randParent < params.pCross:
				if checkValidLocation(parent1.locations[i],mapIn,locations_1):
					locations_1.append(parent1.locations[i])
				else:
					locations_1.append(generateRandomLocation(mapIn,locations_1))

			else:
				if checkValidLocation(parent2.locations[i],mapIn,locations_1):
					locations_1.append(parent2.locations[i])
				else:
					locations_1.append(generateRandomLocation(mapIn,locations_1))

		## Evaluate the location for the second child.
		# If it should mutate, then add a random location.
		if randMutate2 > 1 - params.pMutate:
			locations_2.append(generateRandomLocation(mapIn,locations_2))
		# Else, if parent 1 is randomly chosen, make sure that the location it has is 
		# Unoccupied, and then add it. Otherwise, just mutate.
		else:	
			if randParent < params.pCross:
				if checkValidLocation(parent2.locations[i],mapIn,locations_2):
					locations_2.append(parent2.locations[i])
				else:
					locations_2.append(generateRandomLocation(mapIn,locations_2))

			else:
				if checkValidLocation(parent1.locations[i],mapIn,locations_2):
					locations_2.append(parent1.locations[i])
				else:
					locations_2.append(generateRandomLocation(mapIn,locations_2))

	## Now that both lists are populated, make new classes.
	tCreate = time.time() - initTime
	child1 = GeneticChild.fromLocations(mapIn,locations_1,tCreate)
	tCreate = time.time() - initTime
	child2 = GeneticChild.fromLocations(mapIn, locations_2,tCreate)
	# RETURN GeneticChild CLASSES 

	return child1, child2


def geneticStateSearch(originalMap,params):
   
	# In the current state, these values aren't used, since the count is kept globally.
	# However, in the case of extrapolation of this code, it would be useful to have these counts
	# stored.
	iCount = params.iCount
	cCount = params.cCount
	rCount = params.rCount

	firstRun = True
	timeRun = 0.0
	initTime = time.time()

	lastGen = []
	currentGen =[]
	lastScores = []

	while timeRun < params.timeToRun:

		## First Population: Generate random states and save. 		
		if firstRun == True:
			# Generate  k states randomly	
			for i in range(0,params.k):
				tCreate = time.time() - initTime
				lastGen.append(GeneticChild.fromRandom(originalMap,tCreate))
				lastScores.append(lastGen[i].utilVal)
			print 'End of first run:',len(lastScores)
			firstRun = False

		## Rest of the generations:
		else:
			currentGen = []
			toPop = []

			# Sort a list of the last scores, and save the indices that they correspond to.
			if DEBUG_GENETICS:
				print lastScores
			zippedScores = zip(range(0,params.k),lastScores)
			lastScores_save = lastScores[:]
			lastScores_elite = lastScores[0:params.k2]
			zippedScores.sort(key=lambda x: x[1])
			lastScores = []

			## Elitism: Save the k2 most fit states.	
			#print 'SAVE'
			inds_elite = range(0,params.k2)
			
			for i in range(1,params.k2+1):

				ind_elite = (zippedScores[params.k-i])[0]
				greaterThanElite =   lastScores_elite < lastScores_save[ind_elite]
				
				#print type(greaterThanElite), greaterThanElite
				if (ind_elite < k2):
					lastScores.append((zippedScores[params.k-i])[1])
					currentGen.append(lastGen[ind_elite])
				elif (type(greaterThanElite) != bool) and (any(greaterThanElite)):
					print "replacing"
					lastScores.append((zippedScores[params.k-i])[1])
					currentGen.append(lastGen[ind_elite])
                                                
				else:
					lastScores.append(lastScores_elite[0])
					currentGen.append(lastGen[inds_elite[0]])
					del inds_elite[0]
					del lastScores_elite[0]
				

			if DEBUG_GENETICS:
				print 'Time Found: ',currentGen[0].timeFound
                        
			## Culling: remove the N least fit states.
			#print 'POP'
			for i in range(0,params.numCull):
				toPop.append((zippedScores[i])[0])
			for index in sorted(toPop,reverse=True):
				del lastGen[index]
				del lastScores_save[index]

			# Recreate zipped list for crossover
			zippedScores = zip(range(0,params.k-params.numCull),lastScores_save)
			zippedScores.sort(key=lambda x:x[1])

			## Crossover:
			#print 'crossover'
			for i in range(0,int(math.ceil((params.k-params.k2)/2))):
				# draw random number to pick states.
				### TODO: MAKE SOMEWHAT WEIGHTED.
				indParent1,indParent2 = random.sample(range(0,params.k-params.k2),2)

				child1, child2 = runCrossover(lastGen[indParent1],lastGen[indParent2],originalMap,params,initTime)
				lastScores.append(child1.utilVal)
				currentGen.append(child1)
				lastScores.append(child2.utilVal)
				currentGen.append(child2)
			
			lastGen = currentGen[:]
				

		# Update the current time
		timeRun = time.time() - initTime
		
	zippedScores = zip(range(0,params.k),lastScores)
	zippedScores.sort(key=lambda x: x[1])
	if (zippedScores[params.k-1])[1] > lastScores[0]:
		return lastGen[(zippedScores[params.k-1])[0]],(zippedScores[params.k-1])[0]
	else:
		return lastGen[0],0

'''
Part 2 genetic testing
'''
random.seed()
pMutate = 0.06
pCross = 0.5
timeToRun = 5
k = 100
k2 = 6 # As of now, k2 must be an even number greater than 0. Both 0 and odd numbers are edge cases that can be dealt with.
numCull = 5 
outputLoc = 'hw1p2_genetic.txt'

originalMap,iCount,cCount,rCount = readFile('sample2.txt')
paramsIn = GeneticParams(iCount,cCount,rCount,pMutate,pCross,timeToRun,k,k2,numCull)
result,ind = geneticStateSearch(originalMap,paramsIn)
if DEBUG_GENETICS:
	print 'Util: ',result.utilVal,' Time: ',result.timeFound,' Index: ',ind
	print result.map

writeFile(outputLoc,result.utilVal,result.map,result.timeFound)
