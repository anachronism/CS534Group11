import numpy
from part2 import * # This import is only until we actually incorporate together.
import random,time,copy,math

IND = 100
COM = 200
RES = 300
TOXIC = 400
SCENIC = 500


## Class containing one map candidate.
class GeneticChild:

	def __init__(self, mapIn,buildCost,locations):
		self.map = copy.deepcopy(mapIn)
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
		# print cls.locations
		# print cls.map
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


def runCrossover(parent1, parent2, mapIn,probMutate,probCross):
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
			if randParent < probCross:
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
			if randParent < probCross:
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


def geneticStateSearch(originalMap,iCount,cCount,rCount,timeToRun):
	k = 100
	k2 = 6	
	numCull = 5 ### Or maybe make it so that it's a threshold
	pMutate = 0.06
	pCross = 0.5

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

			# Sort a list of the last scores, and save the indices that they correspond to.
			print lastScores
			zippedScores = zip(range(0,k),lastScores)
			lastScores_save = lastScores[:] 
			zippedScores.sort(key=lambda x: x[1])
			lastScores = []

			## Elitism: Save the k2 most fit states.	
			#print 'SAVE'
			for i in range(1,k2+1):
				ind_elite = (zippedScores[k-i])[0]
				lastScores.append((zippedScores[k-i])[1])
				# print ((zippedScores[k-i])[1])
				#print lastGen[ind_elite].utilVal
				currentGen.append(lastGen[ind_elite])
			
			## Culling: remove the N least fit states.
			#print 'POP'
			for i in range(0,numCull):
				toPop.append((zippedScores[i])[0])
			

			for index in sorted(toPop,reverse=True):
				del lastGen[index]
				del lastScores_save[index]

			# Recreate zipped list for crossover
			zippedScores = zip(range(0,k-numCull),lastScores_save)
			zippedScores.sort(key=lambda x:x[1])

			## Crossover:
			#print 'crossover'
			for i in range(0,int(math.ceil((k-k2)/2))):
				# draw random number to pick states.
				### TODO: MAKE SOMEWHAT WEIGHTED.
				### THIS IS STILL NOT WORKING
				zipParent1 = abs(random.triangular(-(len(zippedScores)-1),len(zippedScores)-1))
				zipParent2 = abs(random.triangular(-(len(zippedScores)-1),len(zippedScores)-1))

				zipParent1 = int(round(zipParent1))
				zipParent2 = int(round(zipParent2))
				#print zipParent1-zipParent2
				
				# indParent1 = zippedScores[zipParent1][0]
				# indParent2 = zippedScores[zipParent2][0]
				
				indParent1,indParent2 = random.sample(range(0,k-k2),2)

				# print indParent1

				#print indParent1

				child1, child2 = runCrossover(lastGen[indParent1],lastGen[indParent2],originalMap,pMutate,pCross)
				#print child1.map
				lastScores.append(child1.utilVal)
				currentGen.append(child1)
				lastScores.append(child2.utilVal)
				currentGen.append(child2)
				
			lastGen = currentGen[:]
				

		# Update the current time
		timeRun = time.time() - initTime


'''
Part 2 genetic testing
'''
random.seed()
originalMap,iCount,cCount,rCount = readFile('sample2.txt')
geneticStateSearch(originalMap,iCount,cCount,rCount, 10)