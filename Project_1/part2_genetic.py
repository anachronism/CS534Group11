import numpy
import part2 as p2 # This import is only until we actually incorporate together.
import random,time

IND = 100
COM = 200
RES = 300
TOXIC = 400
SCENIC = 500


# randomCrossover()

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
	genLocations = numpy.zeros((k,iCount+cCount+rCount), dtype=(int,2))
	#tmpListAvoid = numpy.ones(iCount+cCount+rCount,dtype=(int,2))

	firstRun = True
	timeRun = 0.0
	initTime = time.time()

	while timeRun < timeToRun:
		# Go through the actual process of generating children and picking and mutation etc
		
		if firstRun == True:
			cnt_avoid = 0
			# Generate  k states randomly	
			for i in range(0,k):
				tmpListAvoid = list(listAvoid)
				for j in range(0,iCount+cCount+rCount):
					valid = False
					while valid == False:
						tmpLocation= (random.randint(0,numRows),random.randint(0,numCols))#genLocations[i,j] 
						# Check location's validity.
						# print tmpLocation
						#print(tmpListAvoid)
						if tmpLocation in tmpListAvoid:
							cnt_avoid = cnt_avoid + 1
							#print('avoiding',cnt_avoid)
							pass
						else:
							# leave the while and add current location to random.
							valid = True
							tmpListAvoid.append(tmpLocation)
							genLocations[i,j] = tmpLocation

			# print(genLocations[1,:])
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
listAvoid = [];
mapIn,iCount,cCount,rCount = p2.readFile('sample2.txt')
toxic_sites = numpy.where(mapIn == TOXIC)
for elt,ind in enumerate(toxic_sites[0]):
	listAvoid.append((toxic_sites[0][ind],toxic_sites[1][ind])) 

print(listAvoid[0][:]) # somehow make this into something that works with the other thing.
geneticStateSearch(mapIn,iCount,cCount,rCount,listAvoid, 1)