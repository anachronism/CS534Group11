#start by placing all structures randomly on the board
#recalculate 1 new board for each structure each board holds the cost of moving one structure to every empty square.
import copy,time,math
import numpy
from random import *
from collections import namedtuple



IND = 100
COM = 200
RES = 300
TOXIC = 400
SCENIC = 500
TOTALSCORE = 0
iCount =0
cCount = 0
rCount = 0
UNBUILTMAP = []
BESTSTATE = []
BESTSCORE = -10000	
iList = []
cList = []
rList = []
DEBUGSTATESCORE = 0
#function takes in map(state) calculates score of the map


DEBUG_GENETICS = 0

## STRUCTS:
# Named Tuple containing genetic search parameters.
GeneticParams = namedtuple("GeneticParams", "iCount cCount rCount pMutate pCross nTournament timeToRun k k2 numCull")

# Class containing one map candidate.
class GeneticChild:
	def __init__(self, mapIn,buildCost,locations,timeFound):
		self.map = mapIn
		#self.buildCost = buildCost
		self.locations = locations
		self.timeFound = timeFound
		#print self.locations
		self.utilVal = calculateStateScore(self.map)[0] - self.buildCost
	# Mutliple inits, depending on if a list of locations is provided or not.
	@classmethod
	def fromRandom(cls, mapIn, timeFound):
     	# Can maybe skip this first step, and just feed mapIn into it.
		cls.map =  copy.deepcopy(mapIn)
		cls.timeFound = copy.deepcopy(timeFound)
		cls.map,cls.buildCost,cls.locations = populateSiteMap(cls.map)
		#print cls.buildCost
		return cls(cls.map,cls.buildCost,cls.locations,cls.timeFound)

	@classmethod
	def fromLocations(cls, mapIn, locations,timeFound):
		cls.map = copy.deepcopy(mapIn)
		cls.locations = locations
		#print cls.locations
		cls.timeFound = copy.deepcopy(timeFound)
		cls.map, cls.buildCost = changeSiteMap(cls.map,cls.locations)
		#\print cls.buildCost
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

## MAIN FUNCTIONS
def runCrossover(parent1, parent2, mapIn,params,initTime):
	# Combine the two randomly.
	locations_1 = []
	locations_2 = []

	for i in range(0,len(parent1.locations)):
		# Check if the location will mutate for either child, and picking parents.  
		randMutate1 = random.random()
		randMutate2 = random.random()
		randParent = random.random()

		# Evaluate the location for the first child.
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

		# Evaluate the location for the second child.
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

	# Now that both lists are populated, make new classes.
	tCreate = time.time() - initTime
	child1 = GeneticChild.fromLocations(mapIn,locations_1,tCreate)
	tCreate = time.time() - initTime
	child2 = GeneticChild.fromLocations(mapIn, locations_2,tCreate)
	
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

		# First Population: Generate random states and save. 		
		if firstRun == True:
			# Generate  k states randomly	
			for i in range(0,params.k):
				tCreate = time.time() - initTime
				lastGen.append(GeneticChild.fromRandom(originalMap,tCreate))
				lastScores.append(lastGen[i].utilVal)
			if DEBUG_GENETICS:
				print 'End of first run:',len(lastScores)
			firstRun = False

		# Rest of the generations:
		else:
			currentGen = []
			toPop = []

			# Sort a list of the last scores, and save the indices that they correspond to.
			if DEBUG_GENETICS:
				print lastScores
			zippedScores = zip(range(0,params.k),lastScores)
			lastScores_save = lastScores[:]
			zippedScores.sort(key=lambda x: x[1])
			lastScores = []

			# Elitism: Save the k2 most fit states.	
			lastScores_elite = lastScores_save[0:params.k2]
			inds_elite = range(0,params.k2)
			
			for i in range(1,params.k2+1):
				# With this sort, the objects that are first looked at have the highest
				# index, which aren't the ones that were saved. If it has the same fitness value as
				# the elite values, don't update. Otherwise, do update. 

				ind_elite = (zippedScores[params.k-i])[0]
				greaterThanElite = lastScores_elite < lastScores_save[ind_elite]
				# If it's an index within the elite, just copy over.
				# This may copy a wrong child eventually, but that's not a bad fail case.
				# It has potential to copy the best result multiple times. 
				if (ind_elite < k2):
					lastScores.append((zippedScores[params.k-i])[1])
					currentGen.append(lastGen[ind_elite])
                                # If it's greater than any of the elements that was saved, save it instead. 
				elif (type(greaterThanElite) != bool) and (any(greaterThanElite)):
					if DEBUG_GENETICS:
						print "replacing"
					lastScores.append((zippedScores[params.k-i])[1])
					currentGen.append(lastGen[ind_elite])

                                # Else, carry over best elite value.                
				else:
					lastScores.append(lastScores_elite[0])
					currentGen.append(lastGen[inds_elite[0]])
					del inds_elite[0]
					del lastScores_elite[0]
				
			if DEBUG_GENETICS:
				print 'Time Found: ',currentGen[0].timeFound
                        
			# Culling: remove the N least fit states.
			for i in range(0,params.numCull):
				toPop.append((zippedScores[i])[0])
			for index in sorted(toPop,reverse=True):
				del lastGen[index]
				del lastScores_save[index]

			# Recreate zipped list for crossover
			zippedScores = zip(range(0,params.k-params.numCull),lastScores_save)
			zippedScores.sort(key=lambda x:x[1])

			# Crossover:
			
			for i in range(0,int(math.ceil((params.k-params.k2)/2))):

				# Using Tournament-based selection.			
				# Find Parent 1
				potentialInds = random.sample(range(0,params.k-params.numCull),k=params.nTournament)
				zippedScores = zip(potentialInds,(lastScores_save[i] for i in potentialInds))
				zippedScores.sort(key = lambda x:x[1])
				indParent1 = zippedScores[params.nTournament-1][0]	

                                # Find parent 2
				potentialInds = random.sample(range(0,params.k-params.numCull),k=params.nTournament)
				zippedScores = zip(potentialInds,(lastScores_save[i] for i in potentialInds))
				zippedScores.sort(key = lambda x:x[1])
				indParent2 = zippedScores[params.nTournament-1][0]	

                                # If the second parent happens to be the same as the first, repeat draw until it isnt.
				while indParent2 == indParent1:
					potentialInds = random.sample(range(0,params.k-params.numCull),k=params.nTournament)
					zippedScores = zip(potentialInds,(lastScores_save[i] for i in potentialInds))
					zippedScores.sort(key = lambda x:x[1])
					indParent2 = zippedScores[params.nTournament-1][0]  

				#indParent1,indParent2 = random.sample(range(0,params.k-params.k2),2)

				child1, child2 = runCrossover(lastGen[indParent1],lastGen[indParent2],originalMap,params,initTime)
				lastScores.append(child1.utilVal)
				currentGen.append(child1)
				lastScores.append(child2.utilVal)
				currentGen.append(child2)

			# Copy created children to become the next old generation
			lastGen = currentGen[:]
				
		# Update the current time
		timeRun = time.time() - initTime

	# If the last generation made a result better than the saved best result, return that.	
	zippedScores = zip(range(0,params.k),lastScores)
	zippedScores.sort(key=lambda x: x[1])
	if (zippedScores[params.k-1])[1] > lastScores[0]:
		return lastGen[(zippedScores[params.k-1])[0]],(zippedScores[params.k-1])[0]
	# Otherwise, just return best saved result.
	else:
		return lastGen[0],0


def calculateStateScore(state):
	global UNBUILTMAP
	columns = len(state)
	rows = len(state[0])

	if DEBUGSTATESCORE:
		print(state)
		print("\n")
	stateScore = 0
	buildScore = 0
	for i in range(columns):
		for j in range(rows):
			#print(i,j)
			if (state[i,j] == IND):
				stateScore =  calcScoreForIND([i,j],state, stateScore)
				buildScore = buildScore + UNBUILTMAP[i,j]
				if DEBUGSTATESCORE:
					print(stateScore)
					print("100",i,j)
			if (state[i,j] == COM):
				stateScore = calcScoreForCOM([i,j],state, stateScore)
				buildScore = buildScore + UNBUILTMAP[i,j]
				if DEBUGSTATESCORE:
					print(stateScore)
					print("200", i,j)
			if (state[i,j] == RES):
				stateScore = calcScoreForRES([i,j],state, stateScore)
				buildScore = buildScore + UNBUILTMAP[i,j]
				if DEBUGSTATESCORE:
					print(stateScore)
					print("300",i,j)
	return [stateScore, buildScore]
	 
#IND scoring
#within 2 toxic -10
#within 2 another ind +3
#within 3 res -5
def calcScoreForIND(INDlocation, state, stateScore):
	nearByBuildings = getStructsWithin(INDlocation, state, 3)
	for j in range(len(nearByBuildings)):
		if(nearByBuildings[j][1]==TOXIC and nearByBuildings[j][0] <= 2):
			stateScore = stateScore - 10
			if DEBUGSTATESCORE:
				print("____","TOXIC","____", stateScore)
		if(nearByBuildings[j][1]==IND and nearByBuildings[j][0] <= 2):
			stateScore = stateScore + 1.5
			if DEBUGSTATESCORE:
				print("____","IND","____", stateScore)
		if(nearByBuildings[j][1]==RES and nearByBuildings[j][0] <= 3):
			stateScore = stateScore - 5
			if DEBUGSTATESCORE:
				print("____","RES","____", stateScore)
	return stateScore
#Res scoring
#within 2 toxic -20
#within 2 scenic +10
#within 3 com +5
#within 3 ind -5
def calcScoreForRES(RESlocation, state, stateScore):
	nearByBuildings = getStructsWithin(RESlocation, state, 3)
	for j in range(len(nearByBuildings)):
		if(nearByBuildings[j][1]==TOXIC and nearByBuildings[j][0] <= 2):
			stateScore = stateScore - 20
		if(nearByBuildings[j][1]==SCENIC and nearByBuildings[j][0] <= 2):
			stateScore = stateScore + 10
		if(nearByBuildings[j][1]==COM and nearByBuildings[j][0] <= 3):
			stateScore = stateScore + 5
	return stateScore
#COM scoring
#within 2 toxic -20
#within 3 res +5
#within com -5 
def calcScoreForCOM(COMlocation, state, stateScore):
	nearByBuildings = getStructsWithin(COMlocation, state, 3)
	for j in range(len(nearByBuildings)):
		if(nearByBuildings[j][1]==TOXIC and nearByBuildings[j][0] <= 2):
			stateScore = stateScore - 20
		if(nearByBuildings[j][1]==COM and nearByBuildings[j][0] <= 2):
			stateScore = stateScore - 2.5
	return stateScore
# Gets numpy list of tuples that contains the locations of toxic waste sites.

def getManhDist(loc1,loc2):
	 if(loc1[0] == loc2[0] and loc1[1] == loc2[1] ):
	 	distance = 1000000
	 else:
	 	distance = abs(loc1[0]-loc2[0])+abs(loc1[1]-loc2[1])
	 return distance

#gets location in state(map) returns list of buildings and their distance from loc1 if the building are within distance (dist) of loc1
def getStructsWithin(loc1, state, dist):
	
	columns = len(state[0])
	rows = len(state)
	nearByBuildings = []
	for j in range(rows):
		for i in range(columns):
			if(getManhDist(loc1, [j,i]) <= dist and state[j,i]> 10):
			#	print(i,j)
				holdDist =getManhDist(loc1, [j,i]) 
				nearByBuildings.append([holdDist, state[j,i]])
	if DEBUGSTATESCORE:
		print(nearByBuildings)
	return nearByBuildings

# Takes a string containing the file location, and returns the location counts and the map stored in the file.
def readFile(fileLoc):
	cnt = 0
	#read in first three line
	with open(fileLoc,'r') as f:
		iCount = f.readline()[0]
		cCount = f.readline()[0]
		rCount = f.readline()[0]
	#read in the map part of the file not fully working last value of every row includes '\n'
	mapOut = []

	with open(fileLoc,'r') as f:
		for i in xrange(3):
			f.next()
		for line in f:
			row = (line.strip('\n')).split(',')

			cnt = cnt+1
			#print(row)
			for index,elt in enumerate(row):
				if elt == 'X' or elt == 'X\r':
					row[index] = TOXIC
				else:
					if elt == 'S' or elt == 'S\r':
						row[index] = SCENIC	 

			#print(list(map(int,row)), len(row))
			mapOut.append(list(map(int,row)))
			
	mapOut = numpy.array(mapOut)

	#print("stats", iCount,cCount,rCount)
	#print(mapOut.shape)
	return [mapOut,int(iCount),int(cCount),int(rCount)]

# Given a string that has the file location, a score, a map, and a time, creates a file containing it.
def writeFile(fileLoc, score, mapIn, timeFound):
	with open(fileLoc,'w') as f:
		f.write('Score achieved: '+ str(score) +'\n')
		f.write('Time solution was found: %.2f Seconds \n'%+ timeFound)
		for row in range(0,mapIn.shape[0]):
			for col in range(0,mapIn.shape[1]):
				if mapIn[row,col] == IND:
					f.write(' I ')
				elif mapIn[row,col] == COM:
					f.write(' C ')
				elif mapIn[row,col] == RES:
					f.write(' R ')
				elif mapIn[row,col] == TOXIC:
					f.write(' X ')
				elif mapIn[row,col] == SCENIC:
					f.write(' S ')
				else:
					f.write(' ' + str(mapIn[row,col]) + ' ')
                                                
			f.write('\n')
		f.close()

def getListOfEmptyLocations(siteMap):
	columns = len(siteMap[0])
	rows = len(siteMap)
	emptySiteList = []
	for j in range(rows):
		for i in range(columns):
			if(siteMap[j,i] <= 10 or siteMap[j,i] == SCENIC): 
				emptySiteList.append([j,i])
	return emptySiteList
#at random populate unbuilt spaces with structures listed in first 3 lines of the data input file
def populateSiteMap(siteMap):
	emptySpaceList = []
	buildingCost = 0
	occupiedLocations = []
	emptySpaceList = getListOfEmptyLocations(siteMap)
	for i in range(iCount):
		emptySpaceCnt = len(emptySpaceList)-1
		randNum = randint(1,emptySpaceCnt)
		location = emptySpaceList[randNum]
		occupiedLocations.append(location)
		buildingCost = buildingCost + siteMap[location[0],location[1]]
		siteMap[location[0],location[1]] = IND
		del emptySpaceList[randNum]

	for i in range(cCount):
		emptySpaceCnt = len(emptySpaceList)-1
		randNum = randint(1,emptySpaceCnt)
		location = emptySpaceList[randNum]
		occupiedLocations.append(location)
		buildingCost = buildingCost + siteMap[location[0],location[1]]
		siteMap[location[0],location[1]] = COM
		del emptySpaceList[randNum]


	for i in range(rCount):
		emptySpaceCnt = len(emptySpaceList)-1
		randNum = randint(1,emptySpaceCnt)
		location = emptySpaceList[randNum]
		occupiedLocations.append(location)
		buildingCost = buildingCost + siteMap[location[0],location[1]]
		siteMap[location[0],location[1]] = RES
		del emptySpaceList[randNum]
	
	return [siteMap,buildingCost,occupiedLocations]

# Fill a map with given locations
def changeSiteMap(siteMap,locationsList):
	buildingCost = 0
	for i in range(0,iCount):
		location = locationsList[i]
		buildingCost = buildingCost + siteMap[location[0],location[1]]
		siteMap[location[0],location[1]] = IND

	for i in range(iCount, iCount + cCount):
		location = locationsList[i]
		buildingCost = buildingCost + siteMap[location[0],location[1]]
		siteMap[location[0],location[1]] = COM

	for i in range(iCount + cCount,iCount + cCount + rCount):
		location = locationsList[i]
		buildingCost = buildingCost + siteMap[location[0],location[1]]
		siteMap[location[0],location[1]] = RES

	return [siteMap,buildingCost]


def getLocationsOfAllBuildings(state):
	buildingParam = []
	stateScore = 0
	columns = len(state)
	rows = len(state[0])
	#print(state)
	stateScore = 0
	
	for i in range(columns):
		for j in range(rows):
			if (state[i,j] == IND): 
				buildingList.append([state[i,j],i,j ])
			if (state[i,j] == RES):
				buildingList.append([state[i,j],i,j ])
			if (state[i,j] == COM):
				buildingList.append([state[i,j],i,j ])
	return buildingList


def moveBuildingThroughMap(movingBuilding, State, bestscore):
	state = copy.deepcopy(State)
	columns = len(state)
	rows = len(state[0])

	#orignal location of the building to be moved
	orig_i = movingBuilding[1]
	orig_j = movingBuilding[2]
	holdI = movingBuilding[1]
	holdJ = movingBuilding[2]
	#bestState = copy.deepcopy(state)
	
	state[orig_i,orig_j] = UNBUILTMAP[orig_i,orig_j]


	bestBuildingLocation = []
	score = []
	
	for i in range(columns):
		for j in range(rows):
			if (state[i,j] < 100 or state[i,j] == 500):
				holdLocationValue = state[i,j]
				state[i,j] = movingBuilding[0]
				score = calculateStateScore(state)
				currentScore = score[0] - score[1]
					
				if (currentScore > bestscore):
					# bestState = copy.deepcopy(state)
					# bestscore = currentScore
					# BESTSCORE = bestscore
					bestscore = currentScore
					holdI = i
					holdJ = j
				state[i,j] = holdLocationValue
	#bestState[holdI,holdJ]
	state[holdI,holdJ] = movingbuilding[0]
	# print("_____________i,j", holdI,holdJ)
	# state[holdI, holdJ] = movingbuilding[0]
	print("BEST SCORE_________________", bestscore)
	
	print("BEST STATE___________________________")
	print(state)

	
	return [state, bestscore]
	
'''
 START OF 'MAIN'
'''


# your code

# (UNBUILTMAP, iCount, cCount, rCount) = readFile("sample2.txt")

# siteMap = copy.deepcopy(UNBUILTMAP)


# [siteMap, buildingCost] = populateSiteMap(siteMap)[0:2]
# BESTSTATE = copy.deepcopy(UNBUILTMAP)

# buildingList = getLocationsOfAllBuildings(siteMap)
# print("__________________________________________", len(buildingList))
algRun = 'Genetic'
iCount = None
cCount = None
rCount = None
if algRun == 'Genetic':
	pMutate = 0.06
	pCross = 0.5
	nTournamentParticipants = 5#15 # A value of 1 here is effectively random sampling.
	timeToRun = 0.5
	k = 100
	k2 = 6 # As of now, k2 must be an even number greater than 0. Both 0 and odd numbers are edge cases that can be dealt with.
	numCull = 5
	inputLoc = 'sample1.txt'
	outputLoc = 'hw1p2_genetic_sample1.txt'

	originalMap,iCount,cCount,rCount = readFile(inputLoc)
	print iCount
	paramsIn = GeneticParams(iCount,cCount,rCount,pMutate,pCross,nTournamentParticipants,timeToRun,k,k2,numCull)
	result,ind = geneticStateSearch(originalMap,paramsIn)
	if DEBUG_GENETICS:
		print 'Util: ',result.utilVal,' Time: ',result.timeFound,' Index: ',ind
		print result.map

	writeFile(outputLoc,result.utilVal,result.map,result.timeFound)

	pass

elif algRun == 'HillClimb':
	start_time = time.time()
	elapsed_time = time.time() - start_time

	cycleCount = 0 
	numberOfRestarts = 25
	#numberOfRestarts = 1e6

	#while (elapsed_time < 2):
	listofScores = []
	siteMap = []
	UNBUILTMAP = []
	holdScore = []
	buildingList = []
	best_Score = -10000

	# (UNBUILTMAP, iCount, cCount, rCount) = readFile("sample2.txt")

	# (siteMap, iCount, cCount, rCount) = readFile("bestValue.txt")

	# print calculateStateScore(siteMap)
	# 100/0

	fileLoc = "sample1.txt"
	seed() # Seed RNG
	(UNBUILTMAP, iCount, cCount, rCount) = readFile(fileLoc)
	buildingList = []
	(siteMap, iCount, cCount, rCount) = readFile(fileLoc)
	siteMap, buildingCost = populateSiteMap(siteMap)[0:2]
	BESTSTATE = copy.deepcopy(siteMap)
	holdScore = calculateStateScore(siteMap)
	BESTSCORE = holdScore[0] - holdScore[1]
	buildingList = getLocationsOfAllBuildings(siteMap)
		

	while (cycleCount < numberOfRestarts):
	#while (elapsed_time < 1):
		cycleCount = cycleCount +1
		elapsed_time = time.time() - start_time
		for i in range(len(buildingList)):
			movingbuilding = buildingList[i]
			[siteMap,best_Score] = moveBuildingThroughMap(movingbuilding, siteMap, best_Score)
			listofScores.append(best_Score)
		if(best_Score > BESTSCORE):
			BESTSTATE = []
			BESTSTATE = copy.deepcopy(siteMap)
			BESTSCORE = best_Score
			bestTime = elapsed_time
		buildingList = []
		(siteMap, iCount, cCount, rCount) = readFile(fileLoc)
		siteMap, buildingCost = populateSiteMap(siteMap)[0:2]
		buildingList = getLocationsOfAllBuildings(siteMap)
		
	print(listofScores)
	print BESTSTATE
	print "\n"
	print BESTSCORE	
	print "\n"
	print bestTime
	columns = len(BESTSTATE)
	rows = len(BESTSTATE[0])

	f1 = open('bestValue.txt', 'w')
	f1.write(str(iCount) + "\n")
	f1.write(str(cCount)+ "\n")
	f1.write(str(rCount) + "\n")

	for i in range(columns):
		for j in range(rows):
			if (j != (rows-1)):
				f1.write(str(BESTSTATE[i,j]) + ",")
			else:
				f1.write(str(BESTSTATE[i,j]) + "\n")
	f1.close()	
	outputLoc = "outputFile.txt"

	writeFile(outputLoc,BESTSCORE, BESTSTATE,bestTime)

	#test = 1/0

	# While loop logic
	#===================
	# 1) Figure out potential state.
	# 2) Calculate building cost
	# 3) Calculate state score
	# 4) 
	# print("INITIAL STATE")
	# print(UNBUILTMAP)
	# print("\n")
	# print("INITIAL SCORE", calculateStateScore(UNBUILTMAP))
	#print(UNBUILTMAP)
	#print("\n")


	#[bestState, buildingList, buildingNum, holdLastScore] =  moveBuildingThroughMap(buildingList, UNBUILTMAP)
	#[bestState, buildingList, buildingNum, holdLastScore] =  moveBuildingThroughMap(buildingList, UNBUILTMAP)

	# print("INITIAL STATE")
	# print(UNBUILTMAP)


	#calculateStateScore(siteMap)    






