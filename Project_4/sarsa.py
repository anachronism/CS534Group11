import argparse
import numpy as np 
import random as rng
from copy import deepcopy
import matplotlib.pyplot as plt
import math

## Class containing all learning functions and parameters.
class SARSA:
    
    def __init__(self, rGoal, rPit, rMove, rGiveup, stepSize,nTrain, epsilon,smartEpsilon,gamma, gridWorld):
        self.rGoal = rGoal
        self.rPit = rPit
        self.rMove = rMove
        self.rGiveup = rGiveup
        self.nTrain = nTrain
        self.epsilon = epsilon
        self.smartEpsilon = smartEpsilon
        self.stepSize = stepSize
        self.gridWorld = gridWorld
        self.gamma = gamma
        self.gridSize = self.gridWorld.shape
        self.Q_table = self.initializeQ()
        self.rewardsPerTrial = []
        
        

    def initializeQ(self):
        global G,P
        rowNum = self.gridSize[0]
        columnNum = self.gridSize[1]
        
        Q_table = np.zeros((rowNum, columnNum), dtype=object)
        for i in range(0, rowNum):
            for j in range(0, columnNum):
                if self.gridWorld[i,j] == -float("inf"):
                    Q_table[i][j] = [-float("inf") for x in range(5)]
                else:
                    Q_table[i][j] = [0 for x in range(5)]
 

        return Q_table

    def epsilonGreedyAction(self, currentLocation):
        randomValue = rng.random()
        
        # if randomValue greater than epsilon go with the action that has largest Q_value else pick random action
        if randomValue > self.epsilon:
            Q_values = self.Q_table[currentLocation[0]][currentLocation[1]]
            action = Q_values.index(max(Q_values)) 
            return action
        else:
            # Randomly pick an action, so long as the action doesn't pass through walls.
            action = rng.randint(0,4)
            while self.Q_table[currentLocation[0]][currentLocation[1]][action] == -float('inf'):
                action = rng.randint(0,4) 
            return action
    

    def takeStep(self,location,action):
        global GIVEUP
        if (action == GIVEUP):
            return location

        P_CORRECT = 0.7
        P_RIGHTTURN = 0.1
        P_LEFTTURN = 0.1
        P_EXTRASTEP = 0.1
        extraStep = False

        # Check if step works correctly or not.
        probDirection = rng.random()
        # Correct step:
        if probDirection < P_CORRECT:
            newDir = action
        # 90 deg right:
        elif probDirection < P_CORRECT + P_RIGHTTURN:
            newDir = (action + 1) % 4
        # 90 deg Left:
        elif probDirection < P_CORRECT+P_RIGHTTURN+P_LEFTTURN:
            newDir = (action - 1) % 4
        # extra step:   
        else: # 1 - probDirection <= 0.1
            newDir = action
            extraStep = True
        
        # Accoutn for extra step
        if (extraStep):
            numStep = 2
        else:
            numStep = 1

        prevLocation = [location[0], location[1]]
        # Take as many steps as needed.
        for steps in range(0, numStep):
            if (newDir == 0):
                nextLocation = [prevLocation[0]-1, prevLocation[1]]
            elif (newDir == 1):
                nextLocation = [prevLocation[0], prevLocation[1]+1]
            elif (newDir == 2):
                nextLocation = [prevLocation[0]+1, prevLocation[1]]
            else:
                nextLocation = [prevLocation[0], prevLocation[1]-1]

            locationValue = self.gridWorld[nextLocation[0], nextLocation[1]]
            if (locationValue == -float('inf')):
                nextLocation = prevLocation
            # As soon as we step into a pit or goal where we take one or two steps,
            # end the function by returning the location
            elif (locationValue == self.rPit) or (locationValue == self.rGoal):
                return nextLocation
            else:
                prevLocation = nextLocation


        return nextLocation
        

    # Helper function that act terminating condition of the SARSA algorithm
    def terminateCondition(self, stateLocation, a):
        x = stateLocation[0]
        y = stateLocation[1]
        return (self.gridWorld[x,y] == self.rPit) or (self.gridWorld[x,y] == self.rGoal) or (a == 4)
            
    # Get random location that isn't a goal, pit, or wall
    def getRandomLocation(self):
        stateNotPicked = 1
        randomLocation = []
        while(stateNotPicked):
            randomLocation = [rng.randint(0,(self.gridSize[0])-1), rng.randint(0,(self.gridSize[1])-1)]
            locationValue = self.rewardFunction(randomLocation)  
            if locationValue != self.rPit and locationValue != self.rGoal and locationValue != -float('inf'):
                stateNotPicked = 0
        return randomLocation

    # Helper function that gets the reward of a given state.
    def rewardFunction(self,stateLocation):
        return self.gridWorld[stateLocation[0],stateLocation[1]]
    
    # Update the Q of the current location based on the current, next S,A
    def UpdateQ(self, s, a, nextS, nextA):

        global TERMINALACTION
        global GIVEUP
        
        # Get current Q
        Q = self.Q_table[s[0]][s[1]][a]

        # If the next action is the termination state, there's no nextQ.
        if nextA != TERMINALACTION:
            nextQ = self.Q_table[nextS[0]][nextS[1]][nextA]
        else:
            nextQ = 0

        alpha = self.stepSize
        gamma = self.gamma

        # If action is giveup, ignore the reward at given state and just use giveUpreward.
        if a  == GIVEUP:
            r = self.rGiveup
        else:
            r = self.rewardFunction(s)

        # If Q is uninitialized, set as reward at current spot.
        if Q == 0:
            newQ = r
        else:
            newQ = Q + alpha*(r + gamma*nextQ - Q)
        
        # Update Q
        self.Q_table[s[0]][s[1]][a] = newQ

        return r
        
    
    # SARSA algorithm
    def runSARSA(self):

        global TERMINALACTION
        global EPSILONRESETLIM
        self.initEpsilon = self.epsilon
        for numTrial in range(0,self.nTrain):
            # Initialize with a random state s.
            initLocation = self.getRandomLocation()
            stateLocation = deepcopy(initLocation)
            self.initEpsilon = self.epsilon
            epsilonCnt = 0
            # Choose action a possible from state s using epsilon-greedy
            action = self.epsilonGreedyAction(stateLocation)

            rewardSum = 0
            runTrial = True
            while (runTrial):
                # Get the next state s' using action a from state s
                # Call takeStep

                # If current step is an end state, make the next action to be terminal.
                if (self.terminateCondition(stateLocation,action)):
                    nextStateLocation = stateLocation
                    nextAction = TERMINALACTION
                # Otherwise take step and next action normally.
                else:
                    # should be getting reward from this step.
                    nextStateLocation = self.takeStep(stateLocation, action)

                    # Choose action a' from s' using epsilon-greedy
                    nextAction = self.epsilonGreedyAction(nextStateLocation)



                # Update Q(s,a) entry of the Q function table using the formula
                reward = self.UpdateQ(stateLocation, action, nextStateLocation, nextAction)

                # Add to reward.
                rewardSum += reward

                # Set next state and next action for the next iteration
                stateLocation = nextStateLocation
                action = nextAction
                # If the next iteration is the terminal state, end this loop.
                if action == TERMINALACTION:
                    runTrial = False

                # Update epsilon.
                if self.smartEpsilon == True:
                    epsilonCnt +=1
                    self.epsilon = self.initEpsilon / epsilonCnt 
                    if self.epsilon < EPSILONRESETLIM:
                        self.epsilon = self.initEpsilon
                        epsilonCnt = 1
            # Add reward to history.
            self.rewardsPerTrial.append(rewardSum)
                
        return self.Q_table


    # Helper function to help us draw the actions when making the outputs
    def drawAction(self, action):
        if (action == 0): return '^'
        elif (action == 1): return '>'
        elif (action == 2): return 'v'
        elif (action == 3): return '<'
        else: return '?'


    # Output function returns the grid of recommended actions from every state of the grid world and
    # the future expected reward for that state under the learned policy.
    def plotAllOutputs(self):
        recommendedActions = np.chararray((self.gridSize[0]-2, self.gridSize[1]-2))
        expectedRewards = np.zeros((self.gridSize[0]-2, self.gridSize[1]-2))
        
        for row in range(1,self.gridSize[0]-1):
            for col in range(1,self.gridSize[1]-1):
                Q_values = self.Q_table[row][col]
                max_Q = max(Q_values)
                action = Q_values.index(max(Q_values)) # Pick best action from Q table.

                expectedRewards[row-1][col-1] = math.ceil(max_Q*100)/100
                
                # Draw best action.
                if (self.rewardFunction([row,col]) == self.rPit):
                    recommendedActions[row-1][col-1] = 'P'
                elif (self.rewardFunction([row,col]) == self.rGoal):
                    recommendedActions[row-1][col-1] = 'G'
                else:
                    recommendedActions[row-1][col-1] = self.drawAction(action)
                
        print expectedRewards
        print recommendedActions


        # Figure 1 plots all recommended actions of each state #####################################
        w = 7
        h = 6
        plt.figure(1, figsize=(w, h))
        tb = plt.table(cellText=recommendedActions, loc=(0,0), cellLoc='center')

        tc = tb.properties()['child_artists']
        for cell in tc: 
            cell.set_height(1.0/recommendedActions.shape[0])
            cell.set_width(1.0/recommendedActions.shape[1])

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("Recommended Actions From Each State")

        
        # Figure 2 plots all future expected rewards of each state #################################
        plt.figure(2, figsize=(w, h))
        tb = plt.table(cellText=expectedRewards, loc=(0,0), cellLoc='center')

        tc = tb.properties()['child_artists']
        for cell in tc: 
            cell.set_height(1.0/expectedRewards.shape[0])
            cell.set_width(1.0/expectedRewards.shape[1])

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title("Expected Future Rewards")

        
        # Figure 3 plots the average rewards for every 50 trials ###################################
        plt.figure(3)
        numTrialsPerInterval = 50
        numIntervals = len(self.rewardsPerTrial) / numTrialsPerInterval
        rewardsReshaped = np.reshape(np.matrix(self.rewardsPerTrial), (numIntervals, numTrialsPerInterval))

        averageRewardList = []
        for i in range(0, rewardsReshaped.shape[0]):
            rewardSum = 0
            for j in range(0, rewardsReshaped.shape[1]):
                reward = rewardsReshaped[i,j]
                rewardSum += reward

            averageReward = rewardSum / numTrialsPerInterval
            averageRewardList.append(averageReward)
            
        plt.plot(averageRewardList)
        plt.title("Average Reward Per Every 50 Trials")

        # Show all three plots
        plt.show()




### MAIN ########################################################################################
    
if __name__ == '__main__':

    ### MAIN:
    # Parser:
    parser = argparse.ArgumentParser(description='''CS 534 Assignment 4.''')
    parser.add_argument('--rGoal',dest='rGoal',nargs=1, type=float, default=[5], help='''
                                                Reward for reaching the goal. Default is 5.
                                                ''')
    parser.add_argument('--rPit',dest='rPit',nargs=1, type=float, default=[-2], help='''
                                                Reward for falling into a pit. Default is -2.
                                                ''')
    parser.add_argument('--rMove',dest='rMove',nargs=1,type=float,default=[-0.1], help = '''
                                                Reward for moving. Default is -0.1.
                                                ''')
    parser.add_argument('--rGiveup',dest='rGiveup',nargs=1,type=float,default=[-3], help = '''
                                                Reward for giving up. Default is -3.
                                                ''')
    parser.add_argument('--nTrain',dest='nTrain',nargs=1,type=int,default=[10000], help = '''
                                                Number of trials to train agent for. Default is 10000.
                                                ''')
    parser.add_argument('--epsilon',dest='epsilon',nargs=1,type=float,default=[0.1], help = '''
                                                Epsilon, for e-greedy exploration. Default is 0.1.
                                                ''')

    parser.add_argument('--gamma',dest='gamma',nargs=1,type=float,default=[1], help = '''
                                                Rate to discount future Q values in update function. Default is 1.
                                                ''')
    parser.add_argument('--stepSize',dest='stepSize',nargs=1,type=float,default=[0.1], help = '''
                                                Step size for Q updating. Default is 0.1.
                                                ''')

    parser.add_argument('--smartEpsilon',dest='smartEpsilon',nargs=1,type=bool,default=[False], help = '''
                                                Smart epsilon update flag. Default is False.
                                                ''')

    args = parser.parse_args()   

    # Constants to make gridworld easier to be parsed.
    X = -float('inf')
    P = args.rPit[0]
    G = args.rGoal[0]
    M = args.rMove[0]

    # Encoding of moves in Q table, other contexts.
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    GIVEUP = 4
    TERMINALACTION = 1000000 # arbitrarily picked number, shouldn't matter what it is. An enumerate of some sort could probably work.
    EPSILONRESETLIM = 1e-6
    # grid world representation.
    # The way this gridworld is written, it indexes backwards (in that lower number goes up, higher number goes down)
    GRIDWORLD = np.matrix([[X,X,X,X,X,X,X,X,X], 
                           [X,M,M,M,M,M,M,M,X],
                           [X,M,M,M,M,M,M,M,X],
                           [X,M,M,P,P,M,M,M,X],
                           [X,M,P,G,M,M,P,M,X],
                           [X,M,M,P,P,P,M,M,X],
                           [X,M,M,M,M,M,M,M,X],
                           [X,X,X,X,X,X,X,X,X]])

    # Initialize a SARSA class object
    sarsa = SARSA(G, P, M, args.rGiveup[0], args.stepSize[0],args.nTrain[0], args.epsilon[0],args.smartEpsilon[0],args.gamma[0], GRIDWORLD)

    # Call updatedQ = SARSA.runSARSA
    updatedQ = sarsa.runSARSA()
    # Use updated Q to get the recommended and total rewards of all possible states
    # and also plots rewards per trial
    sarsa.plotAllOutputs()

    # End of main
