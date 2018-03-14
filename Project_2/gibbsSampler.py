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
   
    def updateNode(self,nIteration):
        self.pastValues.append((nIteration,self.currentValue)) # Append list with tuble containing current value and the number of iterations.
        numProbabilities = len(self.possibleValues)

    
        # Returns a list of length (numProbabilities)
        newProb = self.calculateProbabilities()
        
        randSample = random.random()
        runningSumProb = 0
        for ind,elt in enumerate(self.possibleValues):
            runningSumProb += newProb[ind]
            if randSample <= runningSumProb:
                self.currentValue = ind
                break    
   
    def getProbFromTable(self, parentInds,desiredTargetProb):
        #print parentInds
        if len(parentInds) == 0:
            return self.probTable[desiredTargetProb]
        elif len(parentInds) == 1:
            return self.probTable[parentInds[0]][desiredTargetProb]
        elif len(parentInds) == 2:
            return self.probTable[parentInds[0]][parentInds[1]][desiredTargetProb]
        else: # There are 4 parents
            return self.probTable[parentInds[0]][parentInds[1]][parentInds[2]][parentInds[3]][desiredTargetProb]

    def calculateProbabilities(self):
        global BAYESMAP

        parentStates = []
        for elt in self.parents:
            #print elt
            currentNode = BAYESMAP[elt]
            print currentNode.currentValue, ' - X, elt - ', elt
            parentStates.append(currentNode.currentValue)
        probValues = []

        for ind,elt in enumerate(self.possibleValues):
            probFromParents = self.getProbFromTable(parentStates,ind) 
            probFromChildren = 1

            # Update probFromChildren for each child node.
            for elt2 in self.children:
                childStates = []
                currentChild = BAYESMAP[elt2]
                # Get probability for each child node
                for elt3 in currentChild.parents:
                    currentNode = BAYESMAP[elt3]
                    if currentNode == self: ### CHECK TO MAKE SURE THIS BEHAVES AS EXPECTED.
                        childStates.append(ind)
                    else:
                        childStates.append(currentNode.currentValue)

                probFromChildren = probFromChildren * float(currentChild.getProbFromTable(childStates, currentChild.currentValue)) ### WHY IS THIS A STR
               
            #print type(probFromParents)
           # print probFromChildren
            probValues.append(probFromParents*probFromChildren)

        # NORMALIZE probValues.
        probValues = np.divide(probValues, np.sum(probValues))
        return probValues
        
        # For each option in possibleValues:
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

def converStringToValue(converString, nodeName):
    test = 0

### MAIN:
BAYESMAP = dict()
listPossibleNodes = ["price","location","neighborhood","location","children","size","schools","age"]

### CONSTANTS, FOR READABILITY
# Price:
CHEAP = 0
OK = 1
EXPENSIVE = 2
# Amenities:
LOTS = 0
LITTLE = 1
# Things that are bad or good:
BAD_2OPT = 0
GOOD_2OPT = 1
# Location:
GOOD_LOC = 0
BAD_LOC = 1
UGLY_LOC = 2
# Size:
SMALL = 0
MEDIUM = 1
LARGE = 2
# Age:
OLD = 0
NEW = 1

### INITIALIZING NODE TABLE
################   PRICE NODE

priceFileName = "price.csv"
parents = ["location", "age", "schools", "size"]
children = []
probTable = readPriceTable(priceFileName)
possibleValues = ["cheap","ok","expensive"]
currentValue = random.choice([CHEAP,OK,EXPENSIVE])
priceNode = BayesNode(parents, children, probTable, possibleValues,currentValue) 
BAYESMAP["price"] = priceNode

################   AMENETIES NODE
parents = []
children = ["location"]
probTable = [0.3, 0.7]
possibleValues = ["lots","little"]
currentValue = random.choice([LOTS,LITTLE])
amenetiesNode = BayesNode(parents, children, probTable, possibleValues,currentValue) 
BAYESMAP["ameneties"] = amenetiesNode


################   neighborhood NODE
parents = []
children = ["location", "children"]
probTable = [0.4,0.6]
possibleValues = ["bad", "good"]
currentValue = random.choice([BAD_2OPT,GOOD_2OPT])
neighbNode = BayesNode(parents, children, probTable, possibleValues,currentValue) 
BAYESMAP["neighborhood"] = neighbNode


################   LOCATION NODE
parents = ["amenities", "neighborhood"]
children = ["age", "price"]
probTable = [[[[] for i in range(3)] for i in range(2)] for i in range(2)]
probTable[LOTS][BAD_2OPT][GOOD_LOC] = 0.3  
probTable[LOTS][BAD_2OPT][BAD_LOC] = 0.4  
probTable[LOTS][BAD_2OPT][UGLY_LOC] = 0.3  
probTable[LOTS][GOOD_2OPT][GOOD_LOC] = 0.8  
probTable[LOTS][GOOD_2OPT][BAD_LOC] = 0.15  
probTable[LOTS][GOOD_2OPT][UGLY_LOC] = 0.05  
probTable[LITTLE][BAD_2OPT][GOOD_LOC] = 0.2  
probTable[LITTLE][BAD_2OPT][BAD_LOC] = 0.4  
probTable[LITTLE][BAD_2OPT][UGLY_LOC] = 0.4  
probTable[LITTLE][GOOD_2OPT][GOOD_LOC] = 0.5  
probTable[LITTLE][GOOD_2OPT][BAD_LOC] = 0.35  
probTable[LITTLE][GOOD_2OPT][UGLY_LOC] = 0.15
possibleValues = ["good","bad","ugly"] # good is 0, bad is 1, ugly is 2
currentValue = random.choice([GOOD_LOC,BAD_LOC,UGLY_LOC])
locationNode = BayesNode(parents, children, probTable, possibleValues,currentValue) 
BAYESMAP["location"] = locationNode


################   CHILDREN NODE
parents = ["neighborhood"]
children = ["schools"]
probTable = [[[] for i in range(2)] for i in range(2)] 
probTable[BAD_2OPT][BAD_2OPT] = 0.6  
probTable[BAD_2OPT][GOOD_2OPT] = 0.4  
probTable[GOOD_2OPT][BAD_2OPT] = 0.3  
probTable[GOOD_2OPT][GOOD_2OPT] = 0.7 
possibleValues = ["bad","good"]
currentValue = random.choice([BAD_2OPT,GOOD_2OPT])
childrenNode = BayesNode(parents, children, probTable, possibleValues,currentValue) 
BAYESMAP["children"] = childrenNode


################   SIZE NODE
parents = []
children = ["price"]
probTable = [0.33, 0.34, 0.33]
possibleValues = ["small","medium","large"]
currentValue = random.choice([SMALL,MEDIUM,LARGE])
sizeNode = BayesNode(parents, children, probTable, possibleValues,currentValue) 
BAYESMAP["size"] = sizeNode


################   SCHOOLS NODE
parents = ["children"]
children = ["price"]
probTable = [[[] for i in range(2)] for i in range(2)] 
probTable[BAD_2OPT][BAD_2OPT] = 0.7  
probTable[BAD_2OPT][GOOD_2OPT] = 0.3  
probTable[GOOD_2OPT][BAD_2OPT] = 0.8  
probTable[GOOD_2OPT][GOOD_2OPT] = 0.2  
possibleValues = ["bad","good"]
currentValue = random.choice([BAD_2OPT,GOOD_2OPT])
schoolsNode = BayesNode(parents, children, probTable, possibleValues,currentValue) 
BAYESMAP["schools"] = schoolsNode


################   AGE NODE
parents = ["location"]
children = ["price"]
probTable = [[[] for i in range(2)] for i in range(3)] 
probTable[GOOD_LOC][OLD] = 0.3  
probTable[GOOD_LOC][NEW] = 0.7  
probTable[BAD_LOC][OLD] = 0.6  
probTable[BAD_LOC][NEW] = 0.4  
probTable[UGLY_LOC][OLD] = 0.9  
probTable[UGLY_LOC][NEW] = 0.1  
possibleValues = []
possibleValues = ["old","new"]
currentValue = random.choice([OLD,NEW])
ageNode = BayesNode(parents, children, probTable, possibleValues,currentValue) 
BAYESMAP["age"] = ageNode


### PARSE INPUTS, CHANGE EVIDENCE NODE VALUES, MAKE SURE TO RANDOM SELECT
inputString = raw_input("Enter the input string: \n")
inputString = inputString.split()

queryNode = inputString[0]

listEvidenceNodes = []
listEvidenceNodesValues = []

i = 1
while "=" in inputString[i]:
    tempEvidenceNode = inputString[i].split("=")
    listEvidenceNodes.append(tempEvidenceNode[0])
    listEvidenceNodesValues.append(tempEvidenceNode[1])
    tempList = BAYESMAP[tempEvidenceNode[0]].possibleValues
    tempIndex = tempList.index(tempEvidenceNode[1])
    print "\n index:", tempIndex, "\n"
    BAYESMAP[tempEvidenceNode[0]].value = tempEvidenceNode[1]    
    i=i+1

updateNumber = int(inputString[i+1])
dropNumber = int(inputString[i+3])

print listEvidenceNodes[0], listEvidenceNodes[1] 
print listEvidenceNodesValues[0],listEvidenceNodesValues[1]
print updateNumber
print dropNumber

### TODO: Populate with evidence node things
# for elt in whateverEvidenceNodesNeedParsing
# Set BAYESMAP[elt].currentValue = whatever value its being set to

nodesToUpdate = np.setdiff1d(listEvidenceNodes,listPossibleNodes)

# Actual looping through, ### TODO: Figure best way to drop the first M current values
BAYESMAP["age"].updateNode(1)

#price schools=good location=ugly -u 10000 -d 0