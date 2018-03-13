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
            return self.probTable[parentInds[0]][desiredTargetProb]
        elif len(parentInds) == 2:
            return self.probTable[parentInds[0]][parentInds[1]][desiredTargetProb]
        else: # There are 4 parents
            return self.probTable[parentInds[0]][parentInds[1]][parentInds[2]][parentInds[3]][desiredTargetProb]

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

### MAIN:
BAYESMAP = dict()
listPossibleNodes = ["price","location","neighb","location","children","size","schools","age"]
### INITIALIZING NODE TABLE
################   PRICE NODE
CHEAP = 0
OK = 1
EXPENSIVE = 2
priceFileName = "price.csv"
parents = ["location", "age", "school", "size"]
children = []
probTable = readPriceTable(priceFileName)
possibleValue = ["cheap","ok","expensive"]
currentValue = random.choice([CHEAP,OK,EXPENSIVE])
priceNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 
BAYESMAP["price"] = priceNode

################   AMENETIES NODE
LOTS = 0
LITTLE = 1
parents = []
children = ["location"]
probTable = [0.3, 0.7]
possibleValue = ["lots","little"]
currentValue = random.choice([LOTS,LITTLE])
amenetiesNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 
BAYESMAP["ameneties"] = amenetiesNode


################   NEIGHB NODE
BAD_2OPT = 0
GOOD_2OPT = 1
parents = []
children = ["location", "children"]
probTable = [0.4,0.6]
possibleValue = ["bad", "good"]
currentValue = random.choice([BAD_2OPT,GOOD_2OPT])
neighbNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 
BAYESMAP["neighb"] = neighbNode


################   LOCATION NODE
GOOD_LOC = 0
BAD_LOC = 1
UGLY_LOC = 2
parents = ["amenities", "neighb"]
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
possibleValue = ["good","bad","ugly"] # good is 0, bad is 1, ugly is 2
currentValue = random.choice([GOOD_LOC,BAD_LOC,UGLY_LOC])
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
currentValue = random.choice([BAD_2OPT,GOOD_2OPT])
childrenNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 
BAYESMAP["children"] = childrenNode


################   SIZE NODE
SMALL = 0
MEDIUM = 1
LARGE = 2
parents = []
children = ["price"]
probTable = [0.33, 0.34, 0.33]
possibleValue = ["small","medium","large"]
currentValue = random.choice([SMALL,MEDIUM,LARGE])
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
currentValue = random.choice([BAD_2OPT,GOOD_2OPT])
schoolsNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 
BAYESMAP["schools"] = schoolsNode


################   AGE NODE
OLD = 0
NEW = 1
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
currentValue = random.choice([OLD,NEW])
ageNode = BayesNode(parents, children, probTable, possibleValue,currentValue) 
BAYESMAP["age"] = ageNode


### PARSE INPUTS, CHANGE EVIDENCE NODE VALUES, MAKE SURE TO RANDOM SELECT

listEvidenceNode = [] ### TODO: Populate with evidence node things
# for elt in whateverEvidenceNodesNeedParsing
# Set BAYESMAP[elt].currentValue = whatever value its being set to

nodesToUpdate = np.setdiff1d(listEvidenceNodes,listPossibleNodes)

# Actual looping through, ### TODO: Figure best way to drop the first M current values
