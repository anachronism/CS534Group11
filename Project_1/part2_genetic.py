import numpy
import part2 as p2 # This import is only until we actually incorporate together.
import random,time

IND = 100
COM = 200
RES = 300
TOXIC = 400
SCENIC = 500


# randomCrossover()

class GeneticChild:
     def __init__(self, mapIn,utilVal):
		self.map,self.buildCost,self.locations = p2.populateSiteMap(mapIn)
		self.utilVal = utilVal





'''
Start: k list of random, valid states.
calculate fitness scores for these k states.
pick two states.
execute 
'''



def geneticStateSearch(mapIn,iCount,cCount,rCount,listAvoid,timeToRun):
	k = 100
	k2 = numpy.floor(k/20)	
	numCull = 5 ### Or maybe make it so that it's a threshold
	prob_mutate = 0.06

	numRows = mapIn.shape[0]
	numCols = mapIn.shape[1]

	# Gen locations is a iCount+cCount+rCount x k matrix containing tuples
	### TODO: Eventually make this a class.
	genLocations = numpy.zeros((k,numRows,numCols))
	genScores = numpy.zeros(k)

	tmpListAvoid = numpy.ones(iCount+cCount+rCount,dtype=(int,2))

	firstRun = True
	timeRun = 0.0
	initTime = time.time()

	while timeRun < timeToRun:
		## First Population: Generate random states and save. 		
		if firstRun == True:
			# Generate  k states randomly	
			for i in range(0,k):
				tmpListAvoid = list(listAvoid)

				# Generate random locations
				for j in range(0,iCount+cCount+rCount):
					valid = False
					while valid == False:
						tmpLocation= [random.randint(0,numRows-1),random.randint(0,numCols-1)]#genLocations[i,j]
						if not (tmpLocation in tmpListAvoid):
							# leave the while and add current location to mapping.
							valid = True
							tmpListAvoid.append(tmpLocation)
							
							if j < iCount:
								genLocations[i,tmpLocation[0],tmpLocation[1]] = IND
							elif j < iCount + cCount:
								genLocations[i,tmpLocation[0],tmpLocation[1]] = COM
							else: # if j < iCount + cCount + rCount
								genLocations[i,tmpLocation[0],tmpLocation[1]] = RES
							# print 'not passing'
						else:
							pass #print tmpLocation

				# Get score for the randomly generated maps.
				genScores
			# print(genLocations)
			firstRun = False

		else:
			pass
			# Select k2 most fit states to save
			# cull from last gen
			# Use crossover etc to make k-k2 states
				# weight towards states w/ better fitness.
				# Combining 2 states makes 2 successors.
				# Randomly change some bits in some states.


		# Update the current time
		timeRun = time.time() - initTime


'''
Part 2 genetic testing
'''
random.seed()
originalMap,iCount,cCount,rCount = p2.readFile('sample2.txt')
test = GeneticChild(originalMap,5)

# geneticStateSearch(originalMap,iCount,cCount,rCount,listAvoid, 1)