problem2.py contains code to execute both the genetic and the hill climbing
algorithm. The parameters described below allow the user to determine how
the program will run. Please adjust the parameters to match your needs before
running the program. 

algRun = 'Both'

Options for algRun are 'HillClimb', 'Genetic', or 'Both'
If algRun is equal to 'Genetic' only the genetic algorithm will be executed
If algRun is equal to 'HillClimb' only the genetic algorithm will be executed
If algRun is equal to 'Both' genetic AND HillClimg algorithms will be executed

inputLoc = 'sample2.txt'
inputLoc is name of the file that problem2.py takes as input. The file contains
the city map and quantity of buildings to be built

outputLoc_hillClimb = "HillClimbResult.txt"
outputLoc_genetic = 'GeneticResult.txt'
outputLoc_hillClimb is where the best result found by HillClimb will be stored
as well as the time it took find the best result



listOfTimeSettings sets the time limit on how much each algorithm has to find the 
best answer. The algorithm will run once for each time limit in the list. If you
want the program to execute only once put only one value in the list
listOfTimeSettings = []
listOfTimeSettings = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

number of times to repeat the program execution for every time in listOfTimeSetttings the 
program can execute numberOfCycles
numberOfCycles = 1


# Genetic algorithm parameters.
pMutate = 0.06
pCross = 0.5
nTournamentParticipants = 5#15 # A value of 1 here is effectively random sampling.
k = 100
k2 = 2 # As of now, k2 must be an even number greater than 0. Both 0 and odd numbers are edge cases that can be dealt with.
numCull = 50 # This should not be greater than 50, as some configuration of the other parameters will not run. 