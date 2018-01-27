# Heavy N-queens Problem. Queens are heavy and difficult to move.
# Cost: 10 + (number squares traversed)^2.
# Mehods: A* and greedy hill-climbing with restarts.
# Suggested Heuristic in assignment: 
#	* 0 if no attacking queens.
#	* 10 + num attacked queens.
#	* Bonus points if prove if it's an admissible heuristic or not.
import argparse


def a_star(nQueens,boardState):
	# TODO: Actually implement A*
	print('A* will go here')

def hill_climb(nQueens,boardState):
	# TODO: Actually implement hill_climb
	print('hill climbing will go here')


#def assignment_1a(n_queens,type_search):


parser = argparse.ArgumentParser(description='''CS 534 Assignment 1 Part 1.''')
parser.add_argument('--n',dest='nQueens',nargs=1, type=int, default=8, help='''
											Number of queens.
											''')
parser.add_argument('--alg',dest='algMethod',nargs=1, type=int, default=1,choices=[1,2], help='''
											Algorithm to use. a 1 specifies A*, a 2 specifies greedy hill climbing.
											''')

args = parser.parse_args()
# TODO: Generate Start state based on nQueens.
startState = [None]

if args.algMethod == 1:
	retVals = a_star(args.nQueens,startState)
else:
	retVals = hill_climb(args.nQueens,startState)

# TODO: print return values.


'''
 Should return: start state, 
 				number of nodes expanded,
 				Time to solve puzzle,
 				Effective branch factor,
 				cost to solve puzzle.
 				Sequence of moves needed to solve puzzle.
'''