#start by placing all structures randomly on the board
#recalculate 1 new board for each structure each board holds the cost of moving one structure to every empty square.
import copy
import numpy
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

#function takes in map(state) calculates score of the map
def calculateStateScore(state):
	columns = len(state[0])
	rows = len(state)
	
	stateScore = 0
	
	# iList = []
	# cList = []
	# rList = []
 # 	tList = []
 # 	sList = []
 	
	for j in range(columns):
		for i in range(rows):
			if (state[i,j] == IND):
				stateScore =  calcScoreForIND([i,j],state, stateScore)
				print("_______", stateScore)
			if (state[i,j] == COM):
				stateScore = calcScoreForCOM([i,j],state, stateScore)
				print("_______", stateScore)
				#print("TOTALSCORE", TOTALSCORE, "_______", calcScoreForCOM([i,j],state, TOTALSCORE))
				#rList.append((i,j))
			if (state[i,j] == RES):
				stateScore = calcScoreForRES([i,j],state, stateScore)
				print("_______", stateScore)
				#print("TOTALSCORE", TOTALSCORE, "_______",calcScoreForRES([i,j],state, TOTALSCORE))
				#rList.append((i,j))

	return stateScore
	#print(len(iList))		
	#print("rlist",rList)

	#print("TESTING", getStructsWithin(iList[0],state,3))
	#state[iList[0][0],iList[0][1]] = state[iList[0][0],iList[0][1]]+1
	 
#IND scoring
#within 2 toxic -10
#within 2 another ind +3
#within 3 res -5
def calcScoreForIND(INDlocation, state, stateScore):
	nearByBuildings = getStructsWithin(INDlocation, state, 3)
	for j in range(len(nearByBuildings)):
		if(nearByBuildings[j][1]==TOXIC and nearByBuildings[j][0] <= 2):
			stateScore = stateScore - 10
	#		print("__________________________", TOTALSCORE)
		if(nearByBuildings[j][1]==IND and nearByBuildings[j][0] <= 2):
			stateScore = stateScore + 1.5
	#		print("__________________________", TOTALSCORE)
		if(nearByBuildings[j][1]==RES and nearByBuildings[j][0] <= 2):
			stateScore = stateScore - 5
	#		print("__________________________", TOTALSCORE)
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
			if(getManhDist(loc1, [j,i]) <= dist and state[j,i]> 0):
			#	print(i,j)
				holdDist =getManhDist(loc1, [j,i]) 
				nearByBuildings.append([holdDist, state[j,i]])
	
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

def getListOfEmptyLocations(siteMap):
	columns = len(siteMap[0])
	rows = len(siteMap)
	emptySiteList = []
	for j in range(rows):
		for i in range(columns):
			if(siteMap[j,i] <= 10):
				emptySiteList.append([j,i])
	return emptySiteList
#at random populate unbuilt spaces with structures listed in first 3 lines of the data input file
def populateSiteMap(siteMap):
	emptySpaceList = []
	buildingCost = 0
	emptySpaceList = getListOfEmptyLocations(siteMap)
	for i in range(iCount):
		emptySpaceCnt = len(emptySpaceList)-1
		randNum = randint(1,emptySpaceCnt)
		location = emptySpaceList[randNum]
		buildingCost = buildingCost + siteMap[location[0],location[1]]
		siteMap[location[0],location[1]] = IND
		del emptySpaceList[randNum]

	for i in range(cCount):
		emptySpaceCnt = len(emptySpaceList)-1
		randNum = randint(1,emptySpaceCnt)
		location = emptySpaceList[randNum]
		buildingCost = buildingCost + siteMap[location[0],location[1]]
		siteMap[location[0],location[1]] = COM
		del emptySpaceList[randNum]


	for i in range(rCount):
		emptySpaceCnt = len(emptySpaceList)-1
		randNum = randint(1,emptySpaceCnt)
		location = emptySpaceList[randNum]
		buildingCost = buildingCost + siteMap[location[0],location[1]]
		siteMap[location[0],location[1]] = RES
		del emptySpaceList[randNum]
	
	return [siteMap,buildingCost]
'''
 START OF 'MAIN'
'''

Matrix2 = numpy.zeros((6, 5))



(UNBUILTMAP, iCount, cCount, rCount) = readFile("sample2.txt")

siteMap = copy.deepcopy(UNBUILTMAP)
(siteMap, buildingCost) = populateSiteMap(siteMap)

calculateStateScore(siteMap)
print(siteMap)
print("\n")



#calculateStateScore(siteMap)    





