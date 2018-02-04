import numpy
import part2 as p2 # This import is only until we actually incorporate together.
import random

IND = 100
COM = 200
RES = 300
TOXIC = 400
SCENIC = 500


# generateRandomState()
# randomCrossover()


'''
Start: k list of random, valid states.
calculate fitness scores for these k states.
pick two states.
execute 
'''

def geneticStateSearch(mapIn,iCount,cCount,rCount, timeRun):
	k = 100
	k2 = floor(k/20)	
	numCull = 5 ### Or maybe make it so that it's a threshold
	prob_mutate = 0.06

	numRows = mapIn.shape[0]
	numCols = mapIn.shape[1]



'''
Part 2 genetic testing
'''
random.seed()

mapIn,iCount,cCount,rCount = p2.readFile('sample2.txt')
