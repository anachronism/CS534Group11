# Class containing one map candidate.
import random
import numpy as np
import sys
import os
class BayesNode:
    def __init__(self, parents, children,probTable,possibleValues,currentValue):
        self.parents = parents # list of names of parents NOTE: MUST BE IN SAME ORDER AS TABLE.
        self.children = children # list of names of children
        self.probTable = probTable # CPT for this node
        self.possibleValues = possibleValues # list of value names.
        self.currentValue = currentValue  
        self.pastValues = [] 
        self.probSave = []### TODO: REMOVE
   
    def repeatValue(self,nIteration):
        self.pastValues.append((nIteration,self.currentValue))

    def updateNode(self,nIteration):
        self.pastValues.append((nIteration,self.currentValue)) # Append list with tuble containing current value and the number of iterations.
        numProbabilities = len(self.possibleValues)

    
        # Returns a list of length (numProbabilities)
        newProb = self.calculateProbabilities()
        self.probSave.append((newProb))

        randSample = random.random()
        # print 'rand:',randSample, 'probs:', newProb
        runningSumProb = 0
        for ind,elt in enumerate(self.possibleValues):
            runningSumProb += newProb[ind]
            if randSample <= runningSumProb:
                self.currentValue = ind
                # print 'Changing new val'
                break    
   
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
    def calculateProbabilities(self):
        global BAYESMAP

        parentStates = []
        for elt in self.parents:
            #print elt
            currentNode = BAYESMAP[elt]
            #print currentNode.currentValue, ' - X, elt - ', elt
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

                ######
                probFromChildren = probFromChildren * currentChild.getProbFromTable(childStates, currentChild.currentValue) 
               
            #print type(probFromParents)
            # print 'c:', probFromChildren,' p:', probFromParents
            probValues.append(probFromParents*probFromChildren) ### TODO: FIGURE OUT WHY THIS DOESN"T SEEM TO CHANGE.

        # NORMALIZE probValues.
        probValues = np.divide(probValues, np.sum(probValues))
        #print 'prob: ',probValues
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

def converStringToValue(converString, nodeName):
    test = 0

def printBayesMapStatus():
    for elt in listPossibleNodes:
       sys.stdout.write(str(BAYESMAP[elt].currentValue) + ' ')
    sys.stdout.write('\n')

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
BAYESMAP = dict()
listPossibleNodes = ["price","amenities","neighborhood","location","children","size","schools","age"]

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
BAYESMAP["amenities"] = amenetiesNode


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

#parametrs for automated program runs(to perform a single run using user input set automatedState = 0 )
#if automatedState is set to 1:
#updateNumber is increased by updateNumberStep and the program executes until updateNumber is greater than updateNumberMax then:
#updateNumber is set to updateNumberStatic and dropNumber is increaced by dropNumberStep until dropNumber is greater than dropNumberMax 
automatedState = 1
increaseUpdateNumberFlag = 1
increaseDropNumberFlag = 0
updateNumberStep = 5
updateNumberMax = 500
dropNumberStep = 200
dropNumberMax = 450
updateNumberStatic = 500

#initial updateNumber and drop number(if automatedState set to '0' the values will be overwritten by user input)
updateNumber = 0
dropNumber = 0
droptNumberStr = str(dropNumber)
while increaseDropNumberFlag == 1 or increaseUpdateNumberFlag == 1:

    
    ### PARSE INPUTS, CHANGE EVIDENCE NODE VALUES, MAKE SURE TO RANDOM SELECT
    debug =0
    #inputString = raw_input("Enter the input string: \n")
    inputString =  'price neighborhood=good -u 100000 -d 100'
    
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
        #print "\n index:", tempIndex, "\n"
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
            increaseUpdateNumberFlag = 0
            increaseDropNumberFlag = 0
    if debug:
        print listEvidenceNodes[0], listEvidenceNodes[1] 
        print listEvidenceNodesValues[0],listEvidenceNodesValues[1]
        print updateNumber
        print dropNumber


    nodesToUpdate = np.setdiff1d(listPossibleNodes,listEvidenceNodes)
    if debug:
        for i in nodesToUpdate:
            print "None  evidence node", i


    valueHistory = [[]]

    ### ACTUAL GIBBS RUNS HERE:
    cnt = 0 
    for i in range(0,updateNumber):
        nodeToUpdate = random.choice(nodesToUpdate)
        BAYESMAP[nodeToUpdate].updateNode(i)
        for elt in nodesToUpdate:
            if elt != nodeToUpdate:
                BAYESMAP[elt].repeatValue(i)

        #printBayesMapStatus()
    # printBayesMapStatus()

    ### After gibbs runs, get probability.
    numberOf0 = 0
    numberOf1 = 0
    numberOf2 = 0
    validUpdateCnt = 0

    nodeUpdateCnt = len(BAYESMAP[queryNode].pastValues) 
    print "Number of updates:", updateNumber
    print "Number of drop samples:", dropNumber    
    

    probabilitiesReturn = getProbEst(queryNode,dropNumber,nodeUpdateCnt)
    print  "P(", queryNode, "=", BAYESMAP[queryNode].possibleValues[0], ") =", probabilitiesReturn[0]
    print "P(", queryNode, "=", BAYESMAP[queryNode].possibleValues[1], ") =", probabilitiesReturn[1]
    if len(BAYESMAP[queryNode].possibleValues) == 3:
        summaryString3 = queryNode, BAYESMAP[queryNode].possibleValues[2], probabilitiesReturn[2]
        print "P(", queryNode, "=", BAYESMAP[queryNode].possibleValues[2], ") =", probabilitiesReturn[2]

    len(BAYESMAP[queryNode].pastValues)


    ############STORE TEST RESULTS

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
#print  BAYESMAP[queryNode].pastValues
    #for i in valueHistory:
    #    print "___________", i
    # Actual looping through, ### TODO: Figure best way to drop the first M current values




    #print "current value: ", BAYESMAP["age"].currentValue 
    #BAYESMAP["age"].updateNode(1)
    #print "current value: ", BAYESMAP["age"].currentValue


    #price schools=good location=ugly -u 10000 -d 0