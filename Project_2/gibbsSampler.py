# Import libraries that are used.
import random
import numpy as np
import sys
import os

## Class containing one node. 
## Attributes:
# Parents: List containing the names of the parents, as strings.
# Children: List of names of the children nodes, as strings.
# possibleValues: List containing the names of the potential values the node can take, as strings.
# currentValue: The current value of the node, represented as the index to possibleValues to get the current value out.
# pastValues: A running tally of the values that the node has taken, along with the index that it was taken.
class BayesNode:
    def __init__(self, parents, children,probTable,possibleValues):
        self.parents = parents # list of names of parents NOTE: MUST BE IN SAME ORDER AS TABLE.
        self.children = children # list of names of children
        self.probTable = probTable # CPT for this node
        self.possibleValues = possibleValues # list of value names.
        self.currentValue = random.choice(range(0,len(possibleValues))) # choose a random index of the possible values as the initial state.
        self.pastValues = [] 
   
## Methods of BayesNode:   
    # updateNode(self,nIteration). Given the current timestamp provided, update the state in the node. 
    def updateNode(self,nIteration):
        # Save the current value before updating it.
        self.pastValues.append((nIteration,self.currentValue)) 
        
        # Get the new set of probabilities based on the current state of the Bayesmap 
        newProb = self.calculateProbabilities()

        # draw a sample from U(0,1), and chose the state that it corresponds to based on the new probability calculation.
        randSample = random.random()
        runningSumProb = 0
        for ind,elt in enumerate(self.possibleValues):
            # RunningSumProb keeps the running CDF of the probability.
            runningSumProb += newProb[ind]
            if randSample <= runningSumProb:
                self.currentValue = ind
                break    
   
   #calculateProbabilities() With the bayesmap in its current state, update the probability of each possibleValue of thsi node.
    def calculateProbabilities(self):
        global BAYESMAP

        # Get the states of each of the parent nodes.
        parentStates = []
        for elt in self.parents:
            currentNode = BAYESMAP[elt]
            parentStates.append(currentNode.currentValue)
        
        probValues = []
        # For each possible value the current node can take, look up the required probabilities.
        for ind,elt in enumerate(self.possibleValues):
            # Look up p(self == ind | parentStates).
            probFromParents = self.getProbFromTable(parentStates,ind)

            # Initialize probability of the children. 
            probFromChildren = 1
            
            # Update probFromChildren for each child node. Making the assumption that children nodes are conditionally independent,
            # and can be multiplied together.
            for elt2 in self.children:

                childStates = []
                currentChild = BAYESMAP[elt2]
                # Get probability for each child node. Need to check the state of each of the child's parents
                for elt3 in currentChild.parents:
                    currentNode = BAYESMAP[elt3]
                    # If currentNode is the node getting updated, use the possibleValue we are looking at as its state instead of its currentValue.
                    if currentNode == self: 
                        childStates.append(ind)
                    else:
                        childStates.append(currentNode.currentValue)

                # After each child's probability is calculated, update the total probability from the children nodes.
                probFromChildren = probFromChildren * currentChild.getProbFromTable(childStates, currentChild.currentValue) 
               
            # Add the probability of self.possibleValues to the probabilities.
            probValues.append(probFromParents*probFromChildren)

        # NORMALIZE probValues so they sum to 1.
        probValues = np.divide(probValues, np.sum(probValues))
        return probValues

    # getProbFromTable(parentInds,desiredTargetProb): A utility function to look through a table based on the states of the parents, and the desired
    # value to look up.
    def getProbFromTable(self, parentInds,desiredTargetProb):
        if len(parentInds) == 0:
            retVal = self.probTable[desiredTargetProb]
            if retVal == []:
                print 'BUG0'
        elif len(parentInds) == 1:
            retVal = self.probTable[parentInds[0]][desiredTargetProb]
            if retVal == []:
                print 'BUG1'
        elif len(parentInds) == 2:
            retVal = self.probTable[parentInds[0]][parentInds[1]][desiredTargetProb]
            if retVal == []:
                print 'BUG2'
        else: # There are 4 parents
            retVal = self.probTable[parentInds[0]][parentInds[1]][parentInds[2]][parentInds[3]][desiredTargetProb]
            if retVal == []:
                print 'BUG4: ',parentInds
                #print self.probTable

        return retVal

    # repeatValue: When another node is updated other than this one, this function saves the fact that the node did not change.
    def repeatValue(self,nIteration):
        self.pastValues.append((nIteration,self.currentValue))


# readPriceTable(fileLoc): Price table was stored as CSV instead of explicitly input, and so needs to be read in.
# we assume that this csv is in the same directory as the file.
def readPriceTable(fileLoc):
    cnt = 0
   
    probTable = [[[[[[] for i in range(3)] for i in range(3)] for i in range(3)]for i in range(3)]for i in range(3)]
  
    with open(fileLoc,'r') as f:
        for line in f:
            if cnt > 1:
                row = (line.strip('\n')).split(',')
                cnt = cnt+1
                #for index,elt in enumerate(row):
                for i in range(0,3):
                 
                    probTable[int(row[0])][int(row[1])][int(row[2])][int(row[3])][i] = float(row[i+4])
                    if debugMode:
                        print "Value___",probTable[int(row[0])][int(row[1])][int(row[2])][int(row[3])][i]
                    #    
            cnt = cnt + 1        
            
    return probTable

# printBayesMapStatus(): Debug function that prints out the current state of each node in the map.
def printBayesMapStatus():
    for elt in listPossibleNodes:
       sys.stdout.write(str(BAYESMAP[elt].currentValue) + ' ')
    sys.stdout.write('\n')

# getProbEst(nodeToQuery,dropNum,nodeUpdateNum): Given a node to look at, calculate the current estimate of the probability.
def getProbEst(nodeToQuery,dropNum, nodeUpdateNum):
    global BAYESMAP 
    numberOf0 = 0
    numberOf1 = 0
    numberOf2 = 0
    validUpdateCnt = 0
    for i in range(0,nodeUpdateNum):
        if(BAYESMAP[nodeToQuery].pastValues[i][0] >= dropNumber):
            #print BAYESMAP[nodeToQuery].pastValues[i][0]
            validUpdateCnt += 1       
            value = BAYESMAP[queryNode].pastValues[i][1]
            if value == 0:
                numberOf0 += 1
            elif value == 1:
                numberOf1 += 1
            elif value == 2:
                numberOf2 += 1

    if validUpdateCnt > 0:
        probability0 = float(numberOf0)/float(validUpdateCnt)
        probability1 = float(numberOf1)/float(validUpdateCnt)
        if len(BAYESMAP[nodeToQuery].possibleValues) == 3:
            probability2 = float(numberOf2)/float(validUpdateCnt)
            return probability0,probability1,probability2
        else:
            return probability0, probability1
    else:
        probability0 = 0
        probability1 = 0
        if len(BAYESMAP[nodeToQuery].possibleValues) == 3:
            probability2 = 0
            return probability0,probability1,probability2
        else:
            return probability0, probability1


### MAIN:
debugMode = 0

## CONSTANTS, FOR READABILITY
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


#parameters for automated program runs(to perform a single run using user input set automatedState = 0 ). This was used to create the plots we were trying to.
#if automatedState is set to 1:
#updateNumber is increased by updateNumberStep and the program executes until updateNumber is greater than updateNumberMax then:
#updateNumber is set to updateNumberStatic and dropNumber is increaced by dropNumberStep until dropNumber is greater than dropNumberMax 
automatedState = 0
increaseUpdateNumberFlag = 1
increaseDropNumberFlag = 0
updateNumberStep = 500
updateNumberMax = 50000
dropNumberStep = 200
dropNumberMax = 0
updateNumberStatic = 1000

#initial updateNumber and drop number(if automatedState set to '0' the values will be overwritten by user input)
updateNumber = 0
dropNumber = 500
droptNumberStr = str(dropNumber)
while increaseDropNumberFlag == 1 or increaseUpdateNumberFlag == 1:
    # Dictionary that contains all the nodes in the map. 
    BAYESMAP = dict()
    # List of nodes that are possible.
    listPossibleNodes = ["price","amenities","neighborhood","location","children","size","schools","age"]


    ### INITIALIZING NODE TABLE
    ################   PRICE NODE
    priceFileName = "price.csv"
    parents = ["location", "age", "schools", "size"]
    children = []
    probTable = readPriceTable(priceFileName)
    possibleValues = ["cheap","ok","expensive"]
    priceNode = BayesNode(parents, children, probTable, possibleValues) 
    BAYESMAP["price"] = priceNode

    ################   AMENETIES NODE
    parents = []
    children = ["location"]
    probTable = [0.3, 0.7]
    possibleValues = ["lots","little"]
    amenetiesNode = BayesNode(parents, children, probTable, possibleValues) 
    BAYESMAP["amenities"] = amenetiesNode


    ################   neighborhood NODE
    parents = []
    children = ["location", "children"]
    probTable = [0.4,0.6]
    possibleValues = ["bad", "good"]
    neighbNode = BayesNode(parents, children, probTable, possibleValues) 
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
    locationNode = BayesNode(parents, children, probTable, possibleValues) 
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
    childrenNode = BayesNode(parents, children, probTable, possibleValues) 
    BAYESMAP["children"] = childrenNode


    ################   SIZE NODE
    parents = []
    children = ["price"]
    probTable = [0.33, 0.34, 0.33]
    possibleValues = ["small","medium","large"]
    sizeNode = BayesNode(parents, children, probTable, possibleValues) 
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
    schoolsNode = BayesNode(parents, children, probTable, possibleValues) 
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
    ageNode = BayesNode(parents, children, probTable, possibleValues) 
    BAYESMAP["age"] = ageNode

    
    debug = 0

    # Parse input.
    #inputString = 'age location=good -u 10000 -d 100'
    inputString = raw_input("Enter the input string: \n")
    inputString = inputString.split() # split string up by spaces.

    queryNode = inputString[0]
    listEvidenceNodes = []
    listEvidenceNodesValues = []
    i = 1

    # While there are still equals, there are still evidence nodes to read in. 
    while "=" in inputString[i]:
        tempEvidenceNode = inputString[i].split("=")
        listEvidenceNodes.append(tempEvidenceNode[0])
        listEvidenceNodesValues.append(tempEvidenceNode[1])
        tempList = BAYESMAP[tempEvidenceNode[0]].possibleValues
        tempIndex = tempList.index(tempEvidenceNode[1])
        BAYESMAP[tempEvidenceNode[0]].currentValue = tempIndex   
        i += 1

    if automatedState == 0:
        updateNumber = int(inputString[i+1])
        dropNumber = int(inputString[i+3])
        increaseDropNumberFlag = 0 
        increaseUpdateNumberFlag = 0

    else:
        if updateNumber > (updateNumberMax - updateNumberStep):
            increaseUpdateNumberFlag = 0
            increaseDropNumberFlag = 1
        
        if increaseUpdateNumberFlag:
            updateNumber += updateNumberStep
        elif increaseDropNumberFlag:
            updateNumber = updateNumberStatic
            dropNumber += dropNumberStep

        if dropNumber > (dropNumberMax - dropNumberStep) :
            increaseDropNumberFlag = 0

    if debug:
        print listEvidenceNodes[0], listEvidenceNodes[1] 
        print listEvidenceNodesValues[0],listEvidenceNodesValues[1]
        print updateNumber
        print dropNumber
   
    # Get the nodes that need to be updated; eg. nodes that aren't evidence nodes.
    nodesToUpdate = np.setdiff1d(listPossibleNodes,listEvidenceNodes)
    if debug:
        for i in nodesToUpdate:
            print "None  evidence node", i


    for i in range(0, len(listPossibleNodes)):
        nodeName = listPossibleNodes[i]
        BAYESMAP[nodeName].pastValues = []
        BAYESMAP[nodeName]
    
    ### ACTUAL GIBBS RUNS HERE:
    #ORDERED NODE SELCTION
    # nodeIndex = 0
    # for i in range(0,updateNumber):
    #     nodeIndex += 1
    #     if(nodeIndex ==  len(nodesToUpdate)):
    #         nodeIndex = 0
    #     nodeToUpdate = nodesToUpdate[nodeIndex]
    #     BAYESMAP[nodeToUpdate].updateNode(i)
    #     for elt in nodesToUpdate:
    #         if elt != nodeToUpdate:
    #             BAYESMAP[elt].repeatValue(i)
    for i in range(0,updateNumber):
        nodeToUpdate = random.choice(nodesToUpdate)
        BAYESMAP[nodeToUpdate].updateNode(i)
        for elt in nodesToUpdate:
            if elt != nodeToUpdate:
                BAYESMAP[elt].repeatValue(i)

    ### After gibbs runs, get probability.
    nodeUpdateCnt = len(BAYESMAP[queryNode].pastValues) 
    print "Number of updates:", updateNumber
    print "Number of drop samples:", dropNumber    

    probabilitiesReturn = getProbEst(queryNode,dropNumber,nodeUpdateCnt)
    print  "P(", queryNode, "=", BAYESMAP[queryNode].possibleValues[0], ") =", probabilitiesReturn[0]
    print "P(", queryNode, "=", BAYESMAP[queryNode].possibleValues[1], ") =", probabilitiesReturn[1]
    if len(BAYESMAP[queryNode].possibleValues) == 3:
        summaryString3 = queryNode, BAYESMAP[queryNode].possibleValues[2], probabilitiesReturn[2]
        print "P(", queryNode, "=", BAYESMAP[queryNode].possibleValues[2], ") =", probabilitiesReturn[2]

    ### STORE TEST RESULTS IN CSV: Used to create the plots in the assignment.
    resultFileName = '5_resultSummary_' + queryNode + "_" + droptNumberStr + '.csv'
    if os.path.exists(resultFileName):
        resultFile = open(resultFileName, 'a')
        if len(BAYESMAP[queryNode].possibleValues) == 3:
            resultFile.write(queryNode + "," + str(dropNumber) + "," + str(updateNumber) + "," + str(probabilitiesReturn[0]) + "," + str(probabilitiesReturn[1]) + "," + str(probabilitiesReturn[2])  +"\n")
        else:
            resultFile.write(queryNode + "," + str(dropNumber) + "," + str(updateNumber) + "," + str(probabilitiesReturn[0]) + "," + str(probabilitiesReturn[1]) + "\n")    
    else: 
        header = "Query Node Name," + "d," + "u," + BAYESMAP[queryNode].possibleValues[0] +"," + BAYESMAP[queryNode].possibleValues[1]
        resultFile = open(resultFileName, 'a')
        if len(BAYESMAP[queryNode].possibleValues) == 3:
            header += "," + BAYESMAP[queryNode].possibleValues[2]  + "\n"
            resultFile.write(header)
            resultFile.write(queryNode + "," + str(dropNumber) + "," + str(updateNumber) + "," + str(probabilitiesReturn[0]) + "," + str(probabilitiesReturn[1]) + "," + str(probabilitiesReturn[2]) + "\n")
      
        else:
            header += "\n"
            resultFile.write(header)
            resultFile.write(queryNode + "," + str(dropNumber) + "," + str(updateNumber) + "," + str(probabilitiesReturn[0]) + "," + str(probabilitiesReturn[1]) + "\n")    


resultFile.close()
