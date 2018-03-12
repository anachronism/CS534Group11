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



### MAKE NETWORK MAP GLOBAL
BAYESMAP = dict
### GLOBAL, TABLES OF POTENTIAL VALUES.
## A table for each cpt.

