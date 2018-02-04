#start by placing all structures randomly on the board
#recalculate 1 new board for each structure each board holds the cost of moving one structure to every empty square.


IND = 100
COM = 200
RES = 300
TOXIC = 400
SCENIC = 500

import numpy

Matrix2 = numpy.zeros((6, 5))

#function takes in map(state) calculates score of the map
def calculateStateScore(state):
	columns = len(state[0])
	rows = len(state)
	print(state)
	#print(columns,rows)
	totalScore=0
	
	iList = []
	cList = []
	rList = []
 	tList = []
 	sList = []

	for j in range(columns):
		for i in range(rows):
			if (state[i,j] == IND):
				iList.append((i,j))
			if (state[i,j] == COM):
				rList.append((i,j))
			if (state[i,j] == RES):
				cList.append((i,j))
			if (state[i,j] == TOXIC):
				tList.append((i,j))
			if (state[i,j] == TOXIC):
				sList.append((i,j))
	
	#print(len(iList))		
	#print("rlist",rList)

	print(getINDWithin2(iList[0],state, columns,rows))

# Gets numpy list of tuples that contains the locations of toxic waste sites.

def getManhDist(loc1,loc2):
	 if(loc1[0] == loc2[0] and loc1[1] == loc2[1] ):
	 	distance = 1000000
	 else:
	 	distance = abs(loc1[0]-loc2[0])+abs(loc1[1]-loc2[1])
	 return distance

#gets location in state, size of state (columns, rows) returns list of locations containing building within distance 2 of loc1
def getINDWithin2(loc1, state, columns, rows):
	siteList = []
	b = 0;
	b = 4;
	
	for j in range(rows):
		for i in range(columns):
			if(getManhDist(loc1, [j,i]) <= 2 and state[j,i]):
				siteList.append([j,i])
    
    
	leftCIndex = (loc1[0] - 2) if (loc1[0] - 2) > 0 else 0
	rightCIndex = (loc1[0] + 2) if (loc1[0] + 2) < (rows-1) else (rows-1)
	
	topRIndex = (loc1[1] - 2) if (loc1[1] - 2) > 0 else 0
	bottomRIndex = (loc1[1] + 2) if (loc1[1] + 2) < (columns-1) else (columns-1)
	
	
	print(loc1)
	#siteList = state[leftCIndex:rightCIndex,topRIndex:bottomRIndex]
	return siteList

Matrix2[3,1] = TOXIC
Matrix2[1,1] = IND

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
			print(row)
			for index,elt in enumerate(row):
				if elt == 'X':
					row[index] = TOXIC
				else:
					if elt == 'S':
						row[index] = SCENIC	 

			print(list(map(int,row)), len(row))
			mapOut.append(list(map(int,row)))
			
	mapOut = numpy.array(mapOut)

	print("stats", iCount,cCount,rCount)
	print(mapOut.shape)
	return [mapOut,int(iCount),int(cCount),int(rCount)]

'''
 START OF 'MAIN'
'''
Matrix2[2,1] = IND
Matrix2[5,4] = COM
Matrix2[1,4] = RES
Matrix2[3,2] = TOXIC
Matrix2[2,2] = SCENIC

calculateStateScore(Matrix2)    



cnt = 0
#read in first three line
with open('sample2.txt','r') as f:
	iCount = f.readline()[0]
	cCount = f.readline()[0]
	rCount = f.readline()[0]
#read in the map part of the file not fully working last value of every row includes '\n'
with open('sample2.txt','r') as f:
	for i in xrange(3):
		f.next()
	for line in f:
		row = line.split(',')
		cnt = cnt+1
		#print(row, len(row))

#print("stats", iCount,cCount,rCount)

fileResults = readFile('sample2.txt')

mapIn = fileResults[0];
print(fileResults[0])
print(fileResults[0].shape)

loc1 = [2,5]
loc2 = [3,1]

f = open("sample2.txt","r")
iCount = f.read(1)
cCount = f.read(4)
rCount = f.read(5)

f.close()

#print(iCount,cCount,rCount)


print(getManhDist(loc1,loc2))




