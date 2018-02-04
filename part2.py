#start by placing all structures randomly on the board
#recalculate 1 new board for each structure each board holds the cost of moving one structure to every empty square.


import numpy


Matrix2 = numpy.zeros((6, 5))

#function takes in map(state) calculates score of the map
def calculateStateScore(state):
	columns = len(state[0])
	rows = len(state)
	print(state)
	print(columns,rows)
	totalScore=0
	rList = []
	for j in range(columns):
		for i in range(rows):
			if (state[i,j] == 50):
				iList.append((i,j))
			if (state[i,j] == 60):
				rList.append((i,j))
			if (state[i,j] == 70):
				rList.append((i,j))
			
	print("rlist",rList)


def getManhDist(loc1,loc2):
	 distance = abs(loc1[0]-loc2[0])+abs(loc1[1]-loc2[1])
	 return distance


Matrix2[2,1] = 50
Matrix2[5,4] = 50
Matrix2[1,4] = 60
Matrix2[3,2] = 70
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
		print(row, len(row))

print("stats", iCount,cCount,rCount)


loc1 = [2,5]
loc2 = [3,1]

f = open("sample2.txt","r")
iCount = f.read(1)
cCount = f.read(4)
rCount = f.read(5)

f.close()

#print(iCount,cCount,rCount)


print(getManhDist(loc1,loc2))




