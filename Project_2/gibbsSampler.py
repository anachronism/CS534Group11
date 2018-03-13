# Class containing one map candidate.
import random
import numpy as np
class BayesNode:
    def __init__(self, parents, children,probTable,possibleValues,currentValue):
        self.parents = parents # list of names of parents NOTE: MUST BE IN SAME ORDER AS TABLE.
        self.children = children # list of names of children
        self.probTable = probTable # CPT for this node
        self.possibleValues = possibleValues # list of value names.
        self.currentValue = currentValue  
        self.pastValues = []

    def updateNode(self):
        self.pastValues.append(self.currentValue)
        numProbabilities = len(self.possibleValues)

    
        # Returns a list of length (numProbabilities)
        newProb = self.calculateProbabilities()
        
        randSample = random()
        runningSumProb = 0
        for ind,elt in enumerate(possibleValues):
            runningProb += newProb[ind]
            if randSample <= runningSumProb:
                self.currentValue = ind
                break    

    def getProbFromTable(self, parentInds,desiredTargetProb):
        if len(parentInds) == 0:
            return self.probTable[desiredTargetProb]
        elif len(parentInds) == 1:
            return self.probTable[parentInds,desiredTargetProb]
        elif len(parentInds) == 2:
            return self.probTable[parentInds[0],parentInds[1],desiredTargetProb]
        else: # There are 4 parents
            return self.probTable[parentInds[0],parentInds[1],parentInds[2],parentInds[3],desiredTargetProb]

    def calculateProbabilies(self):
        global BAYESMAP

        parentStates = []
        for elt in self.parents:
            currentNode = BAYESMAP[elt]
            parentStates.append(currentNode.currentValue)

        probValues = [];
        for ind,elt in enumerate(self.possibleValues):
            probFromParents = self.getProbFromTable(parentStates,ind)
            probFromChildren = 1

            # Update probFromChildren for each child node.
            for elt2 in self.children:
                childStates = []
                # Get probability for each child node
                for elt3 in elt2.parents:
                    currentNode = BAYESMAP[elt]
                    if currentNode == self: ### CHECK TO MAKE SURE THIS BEHAVES AS EXPECTED.
                        childStates.append(ind)
                    else:
                        childStates.append(currentNode.currentValue)

                probFromChildren = probFromChildren * self.getProbFromTable(childStates, elt2.currentValue)

            probValues.append(probFromParents*probFromChildren)

        # NORMALIZE probValues.
        probValues = np.divide(probValues, np.sum(probValues))
        return probValues
        
        # For each option in possiblevalues:
            # Calculate P(that result | parent states)
            # Calculate P(child states | all known states including result)
                # This splits up for each child state.
            # Multiply together.

        # Normalize probabilities to sum to 1.
        # Return list of probabilities.

debugMode = 0

def readPriceTable(fileLoc):
    cnt = 0
   
    probTable = [[[[[[] for i in range(3)] for i in range(3)] for i in range(3)]for i in range(3)]for i in range(3)]
  
    with open(fileLoc,'r') as f:
        for line in f:
            if cnt > 2:
                row = (line.strip('\n')).split(',')
                cnt = cnt+1
                #for index,elt in enumerate(row):
                for i in range(0,3):
                 
                    probTable[int(row[0])][int(row[1])][int(row[2])][int(row[3])][i] = row[i+4]
                    if debugMode:
                        print "Value___",probTable[int(row[0])][int(row[1])][int(row[2])][int(row[3])][i]
                    #    
            cnt = cnt + 1        
            
    return probTable


### MAKE NETWORK MAP GLOBAL
BAYESMAP = dict()

################   PRICE NODE
priceFileName = "price.csv"

parents = ["location", "age", "school", "size"]
children = []
probTable = readPriceTable(priceFileName)
possibleValue = ["cheap","ok","expensive"]
currentValue = [0]
 
priceNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 

BAYESMAP["price"] = priceNode


################   AMENETIES NODE
parents = []
children = ["location"]
probTable = [0, 0.3]
possibleValue = ["lots","little"]
currentValue = [0]
 
amenetiesNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 

BAYESMAP["ameneties"] = amenetiesNode


################   NEIGHB NODE
parents = []
children = ["location", "children"]
probTable = [0.4,0.6]
possibleValue = ["bad", "good"]
currentValue = [0]
 
amenetiesNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 

BAYESMAP["neighb"] = amenetiesNode


################   LOCATION NODE
parents = ["amenities", "neighborhood"]
children = ["age", "price"]
probTable = [[[[] for i in range(3)] for i in range(3)] for i in range(3)]
probTable[0][0][0] = 0.3  
probTable[0][0][1] = 0.4  
probTable[0][0][2] = 0.3  
probTable[0][1][0] = 0.8  
probTable[0][1][1] = 0.15  
probTable[0][1][2] = 0.05  
probTable[1][0][0] = 0.2  
probTable[1][0][1] = 0.4  
probTable[1][0][2] = 0.4  
probTable[1][1][0] = 0.5  
probTable[1][1][0] = 0.35  
probTable[1][1][0] = 0.15

possibleValue = ["good","bad","ugly"]
currentValue = [0]
 
locationNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 

BAYESMAP["location"] = locationNode


################   CHILDREN NODE
parents = ["neighb"]
children = ["schools"]
probTable = [[[] for i in range(2)] for i in range(2)] 
probTable[0][0] = 0.6  
probTable[0][1] = 0.4  
probTable[1][0] = 0.3  
probTable[1][1] = 0.7 
possibleValue = ["bad","good"]
currentValue = [0]
 
childrenNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 

BAYESMAP["children"] = childrenNode


################   SIZE NODE
parents = []
children = ["price"]
probTable = [0.33, 0.34, 0.33]
possibleValue = ["small","medium","large"]
currentValue = [0]
 
sizeNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 

BAYESMAP["size"] = sizeNode


################   SCHOOLS NODE
parents = ["children"]
children = ["price"]
probTable = [[[] for i in range(2)] for i in range(2)] 
probTable[0][0] = 0.7  
probTable[0][1] = 0.3  
probTable[1][0] = 0.8  
probTable[1][1] = 0.2  
possibleValue = ["bad","good"]
currentValue = [0]
 
schoolsNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 

BAYESMAP["schools"] = schoolsNode


################   AGE NODE
parents = ["location"]
children = ["price"]
probTable = [[[[] for i in range(2)] for i in range(2)] for i in range(2)] 
probTable[0][0][0] = 0.3  
probTable[0][0][1] = 0.7  
probTable[0][1][0] = 0.6  
probTable[0][1][1] = 0.4  
probTable[1][0][0] = 0.9  
probTable[1][1][1] = 0.1  
possibleValue = ["old","new"]
currentValue = [0]
 
ageNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 

BAYESMAP["age"] = ageNode




### GLOBAL, TABLES OF POTENTIAL VALUES.
## A table for each cpt.

