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
	print(columns,rows)
	totalScore=0
	
	iList = []
	cList = []
	rList = []
 	tList = []
 
	for j in range(columns):
		for i in range(rows):
			if (state[i,j] == 100):
				iList.append((i,j))
			if (state[i,j] == 200):
				rList.append((i,j))
			if (state[i,j] == 300):
				cList.append((i,j))
			if (state[i,j] == 300):
				cList.append((i,j))
	
	print(len(iList))		
	#for j in range(len(iList))
	print("rlist",rList)


def getManhDist(loc1,loc2):
	 distance = abs(loc1[0]-loc2[0])+abs(loc1[1]-loc2[1])
	 return distance

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
			print(row, len(row))
			mapOut.append(row)
			
	mapOut = numpy.array(mapOut)
	print("stats", iCount,cCount,rCount)
	print(mapOut.shape)
	return [mapOut,iCount,cCount,rCount]

'''
 START OF 'MAIN'
'''
Matrix2[2,1] = IND
Matrix2[5,4] = COM
Matrix2[1,4] = RES
Matrix2[3,2] = TOXIC

calculateStateScore(Matrix2)    

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




