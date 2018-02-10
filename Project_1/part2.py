#start by placing all structures randomly on the board
#recalculate 1 new board for each structure each board holds the cost of moving one structure to every empty square.
import copy
import numpy
import time
from random import *

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


start_time = time.time()
elapsed_time = time.time() - start_time

cycleCount = 0 
numberOfRestarts = 25


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

(UNBUILTMAP, iCount, cCount, rCount) = readFile("sample2.txt")
buildingList = []
(siteMap, iCount, cCount, rCount) = readFile("sample2.txt")
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
	(siteMap, iCount, cCount, rCount) = readFile("sample2.txt")
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





