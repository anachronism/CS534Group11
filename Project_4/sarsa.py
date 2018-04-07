import argparse
import numpy as np 
import random as rng


class SARSA:
    
    def __init__(self, rGoal, rPit, rMove, rGiveup, nTrain, epsilon, gridWorld):
        self.rGoal = rGoal
        self.rPit = rPit
        self.rMove = rMove
        self.rGiveup = rGiveup
        self.nTrain = nTrain
        self.epsilon = epsilon
        self.gridWorld = gridWorld
        self.gridSize = [6,7]
        print self.gridSize[0]#.shape[0], gridWorld.shape[1]]
        

    def initializeQ(self):
        Q_table = []
        columnNum = self.gridSize[0]
        rowNum = self.gridSize[1]

        #matrix.append([0,0,0,0,0])
        Q_table = [[0 for x in range(columnNum)] for y in range(rowNum)]
        for i in range(0, rowNum):
            for j in range(0, columnNum):
                Q_table[i][j] = [0 for x in range(5)]

        Q_table[2][2] = [-100, -100, -100, -100, -100]
        Q_table[2][3] = [-100, -100, -100, -100, -100]
        Q_table[3][1] = [-100, -100, -100, -100, -100]
        Q_table[3][5] = [-100, -100, -100, -100, -100]
        Q_table[4][2] = [-100, -100, -100, -100, -100]
        Q_table[4][3] = [-100, -100, -100, -100, -100]
        Q_table[4][4] = [-100, -100, -100, -100, -100]

        Q_table[3][2] = [100, 100, 100, 100, 100]        
        return Q_table

    def epsilonGreedyAction(self):
        print
    def takeStep(self):
        print
        # TODO: Use Max's takeStep function

    # Helper function that act terminating condition of the SARSA algorithm
    def terminateCondidtion(x, y, a):
        print
    def getRandomState(self):
        stateNotPicked = 0
        state = [0, 0]
        while(stateNotPicked):
            state[0,0] = [random.randint(0,gridSize[0]),random.randint(0,gridSize[1])]
            stateValue = self.gridWorld[state[0],state[1]]  
            if stateValue != self.rPit or stateValue != self.rGoal:
                stateNotPicked = 1
        return state
    # SARSA algorithm
    def runSARSA(self):
        self.Q_table = initializeQ()

        for numTrial in range(0,nTrain):
            print
            # Initialize a random state s, choose a random state
            


            # Choose action a possible from state s using epsilon-greedy
            # TODO: Make an epsilon-greedy function to choose next action...?
            # action = epsilonGreedyAction ????
        while (self.terminateCondition(x,y,a)):
            print
                # Get the next state s' using action a from state s
                # Call takeStep
                
                # Choose action a' from s' using epsilon-greedy

                # Update Q(s,a) entry of the Q function table using the formula

                # Set next state and next action for the next iteration
        return self.Q_table


#if if __name__ == '__main__':
    # TODO: Put of all Max's argparse code here

    # Initialize a SARSA class objecy

    # Call updatedQ = SARSA.runSARSA

    # Use updated Q to get the paths and total rewards of all possible states to end states

    # End of main


### MAX: I think that direction should be encoded as [0,1,2,3].
###     If you want to change it then change how this part works.
###     Currently, assuming that the increments move clockwise.
###     Eg. 0 -> N, 1 -> E, 2 -> S, 3 -> W

def takeStep(location,map,direction):
    P_CORRECT = 0.7
    P_RIGHTTURN = 0.1
    P_LEFTTURN = 0.1
    P_EXTRASTEP = 0.1
    extraStep = False

    probDirection = rng.random()
    if probDirection < P_CORRECT:
        newDir = direction
    elif probDirection < P_CORRECT + P_RIGHTTURN:
        newDir = (direction + 1)%4
    elif probDIrection < P_CORRECT+P_RIGHTTURN+P_LEFTTURN:
        newDir = (direction - 1) % 4
    else: # 1 - probDirection <= 0.1
        newDir = direction
        extraStep = True
    ### TODO: ACTUALLY TAKE STEP.


### MAIN:
# Parser:
parser = argparse.ArgumentParser(description='''CS 534 Assignment 4.''')
parser.add_argument('--rGoal',dest='rGoal',nargs=1, type=int, default=5, help='''
                                            Reward for reaching the goal. Default is 5.
                                            ''')
parser.add_argument('--rPit',dest='rPit',nargs=1, type=int, default=-2, help='''
                                            Reward for falling into a pit. Default is -2.
                                            ''')
parser.add_argument('--rMove',dest='rMove',nargs=1,type=int,default=-0.1, help = '''
                                            Reward for moving. Default is -0.1.
                                            ''')
parser.add_argument('--rGiveup',dest='rGiveup',nargs=1,type=int,default=-3, help = '''
                                            Reward for giving up. Default is -3.
                                            ''')
parser.add_argument('--nTrain',dest='nTrain',nargs=1,type=int,default=10000, help = '''
                                            Number of trials to train agent for. Default is 10000.
                                            ''')
parser.add_argument('--epsilon',dest='epsilon',nargs=1,type=int,default=0.1, help = '''
                                            Epsilon, for e-greedy exploration. Default is 0.1.
                                            ''')

args = parser.parse_args()

## OUTPUT:
# A gridworld, where each non terminal state has:
    # Recommended action from that state
    # Future expected reward for that state under learned policy.

# Be careful computing the recommended action from a state; 
# there are multiple reasons you cannot simply select the 
# neighboring state with the highest expected reward.    

# Constants to make gridworld easier to be parsed.
X = -float('inf')
P = args.rPit
G = args.rGoal

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
GIVEUP = 4

### TODO: Not necessarily with this GRIDWORLD object, but when it starts make sure to update the gridworld to have smarter
###         initial values.

GRIDWORLD = ([X,X,X,X,X,X,X,X,X], 
            [X,0,0,0,0,0,0,0,X],
            [X,0,0,0,0,0,0,0,X],
            [X,0,0,P,P,0,0,0,X],
            [X,0,P,G,0,0,P,0,X],
            [X,0,0,P,P,P,0,0,X],
            [X,0,0,0,0,0,0,0,X],
            [X,X,X,X,X,X,X,X,X])


sarsa1 = SARSA(G, P, -1, -5, 1000, 0.01, GRIDWORLD)
print sarsa1.getRandomState()
### TODO:: PSEUDOCODE, PLS UPDATE
