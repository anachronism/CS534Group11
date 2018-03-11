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
            for elt2 in self.children:
                currentNode = BAYESMAP[elt]
                probFromChildren = probFromChildren * self.getProbFromTable(PARENTSOFCHILDREN, currentNode.currentValue)
                ### THIS IS A HARD CASE self.getPRobFromTable(PARENTSOFCHILDREN,childStates[]) Need to get parents of children.
                ### The main issue si knowing where the target node is in the list of the child's parents.
            probValues.append(probFromParents*probFromChildren)
        return probValues
        # For each option in possiblevalues:
            # Calculate P(that result | parent states)
            # Calculate P(child states | all known states including result)
                # This splits up for each child state.
            # Multiply together.

        # Normalize probabilities to sum to 1.
        # Return list of probabilities.



### MAKE NETWORK MAP GLOBAL
BAYESMAP = dict
### GLOBAL, TABLES OF POTENTIAL VALUES.
## A table for each cpt.

