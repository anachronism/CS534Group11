import numpy
from part2 import * # This import is only until we actually incorporate together.
import random,time,copy,math

IND = 100
COM = 200
RES = 300
TOXIC = 400
SCENIC = 500


# randomCrossover()
## Class containing one map candidate.
class GeneticChild:

	def __init__(self, mapIn,buildCost,locations):
		self.map = mapIn
		self.buildCost = buildCost
		self.locations = locations
		self.utilVal = calculateStateScore(self.map) - self.buildCost

	## Mutliple inits, depending on if a list of locations is provided or not.
	@classmethod
	def fromRandom(cls, mapIn):
     	# Can maybe skip this first step, and just feed mapIn into it.
		cls.map =  copy.deepcopy(mapIn)
		cls.map,cls.buildCost,cls.locations = populateSiteMap(cls.map)
		return cls(cls.map,cls.buildCost,cls.locations)

	@classmethod
	def fromLocations(cls, mapIn, locations):
		cls.map = copy.deepcopy(mapIn)
		cls.locations = locations
		cls.map, cls.buildCost = changeSiteMap(cls.map,cls.locations)
		return cls(cls.map,cls.buildCost,cls.locations)


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


def runCrossover(parent1, parent2, mapIn,probMutate):
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
		if randMutate1 > 1 - probMutate:
			locations_1.append(generateRandomLocation(mapIn,locations_1))
		# Else, if parent 1 is randomly chosen, make sure that the location it has is 
		# Unoccupied, and then add it. Otherwise, just mutate.
		else:	
			if randParent < 0.5:
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
		if randMutate2 > 1 - probMutate:
			locations_2.append(generateRandomLocation(mapIn,locations_2))
		# Else, if parent 1 is randomly chosen, make sure that the location it has is 
		# Unoccupied, and then add it. Otherwise, just mutate.
		else:	
			if randParent < 0.5:
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
	child1 = GeneticChild.fromLocations(mapIn,locations_1)
	child2 = GeneticChild.fromLocations(mapIn, locations_2)
	# RETURN GeneticChild CLASSES 

	return child1, child2

'''
Start: k list of random, valid states.
calculate fitness scores for these k states.
pick two states.
execute 
'''



def geneticStateSearch(originalMap,iCount,cCount,rCount,timeToRun):
	k = 100
	k2 = 6	
	numCull = 5 ### Or maybe make it so that it's a threshold
	pMutate = 0.06

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
				lastGen.append(GeneticChild.fromRandom(originalMap))
				lastScores.append(lastGen[i].utilVal)
			print 'End of first run:',len(lastScores)
			firstRun = False
		else:
			currentGen = []
			toPop = []
			# Select k2 most fit states to save
			zippedScores = zip(range(0,k),lastScores) 
			zippedScores.sort(key=lambda x: x[1])
			lastScores = []
			print 'SAVE'
			for i in range(1,k2+1):
				ind_elite = (zippedScores[k-i])[0]
				lastScores.append((zippedScores[k-i])[1])
				print ((zippedScores[k-i])[1])
				#print lastGen[ind_elite].utilVal
				currentGen.append(lastGen[ind_elite])
			
			## 
			# cull from last gen
			print 'POP'
			for i in range(0,numCull):
				toPop.append((zippedScores[i])[0])
			
			for index in sorted(toPop,reverse=True):
				del lastGen[index]

			### TODO: ACTUALLY IMPLEMENT CROSSOVER
			# Use crossover etc to make k-k2 states
			print 'crossover'
			for i in range(0,int(math.ceil((k-k2)/2))):
				# draw random number to pick states.
				### TODO: MAKE SOMEWHAT WEIGHTED.
				indParents = random.sample(range(0,(k-k2)-1),2)
				child1, child2 = runCrossover(lastGen[indParents[0]],lastGen[indParents[1]],originalMap,pMutate)
				
				lastScores.append(child1.utilVal)
				currentGen.append(child1)
				lastScores.append(child2.utilVal)
				currentGen.append(child2)
				# weight towards states w/ better fitness.
				# Combining 2 states makes 2 successors.
				# Randomly change some bits in some states.
			lastGen = copy.deepcopy(currentGen)
			## TODO: UPDATE LAST SCORE
			

		# Update the current time
		timeRun = time.time() - initTime


'''
Part 2 genetic testing
'''
random.seed()
originalMap,iCount,cCount,rCount = readFile('sample2.txt')
# test = GeneticChild.fromRandom(originalMap)
# print test.utilVal
geneticStateSearch(originalMap,iCount,cCount,rCount, 10)