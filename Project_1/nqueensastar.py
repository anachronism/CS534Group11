# Calvin He, Max Li, Ilya Lifshits, Rohit Voleti
# CS 534 Aritificial Intelligence Spring 2018
# Group 11
# Assignment 1 Part 1
# Heavy N Queens with A* Approach

import numpy as np
import matplotlib.pyplot as plt
import random
import heapq
import time
import sys

class NQueensAStar:

    # Constructor and initializing class atributes
    def __init__(self, n):
        self.num_queens = n
        self.start_state = np.zeros((n,n))
        self.queen_value = -1
        self.initialize_location()
        self.frontier = PriorityQueue()
        self.came_from = {}
        self.cost_so_far = {}
        self.expansion_count = 0
        self.start_time = None
        self.end_time = None

        
        
    # Class method initializes positions of the queens with randomization
    def initialize_location(self):
        for col in range(self.num_queens):
            queen_row = random.randint(0,self.num_queens-1)
            self.start_state[queen_row][col] = self.queen_value


            
    # Class method calculates the number of pairs of attacking queens for a specific board state
    def get_attack_queens(self, board):
        # Sort indices of queen so that we consider them one at a time from left to right
        (row_q, col_q) = np.where(board == self.queen_value)
        sorted_indices = np.argsort(col_q)
        col_q_sorted = col_q[sorted_indices]
        row_q_sorted = row_q[sorted_indices]

        num_attack_queens = 0
        row_copy = row_q_sorted
        while (len(row_copy) != 0):
            current_row = row_copy[0]
            for i in range(1, len(row_copy)):
                next_row = row_copy[i]
                # If they are in the same row, then they are attacking
                if next_row == current_row:
                    num_attack_queens += 1
                # If the distance betweem them in rows is the same as the distance between
                # them in columns, then they are attacking each other diagonally
                if abs(next_row - current_row) == i:
                    num_attack_queens += 1

            # Remove the queen to avoid double counting
            row_copy = row_copy[1:]
            
        return num_attack_queens


    
    # Class method that utilizes priority queue, costs, heurisitcs, and successors of the board to execute an A* search
    # Referenced from "Introduction to A*" from the Red Blob Games    
    def a_star_search(self):
        # Start time of A* search
        self.start_time = time.time()

        # An n-d array is not hashable in lists, and can get altered due to being compared to other items in a heap,
        # so a workaround to this is to represent the matrix state as a string.        
        start_state_str = self.start_state.tostring()
        self.frontier.put(start_state_str, 0)
        self.came_from[start_state_str] = None
        self.cost_so_far[start_state_str] = 0

        # While loop handles the expansion of successors in the priority queue
        while not self.frontier.empty():
            # Pop from heap with priority and convert string representation back to a matrix state
            (current_priority, current_state_str) = self.frontier.get()
            current_state = np.fromstring(current_state_str,dtype=float)
            current_state = current_state.reshape(self.num_queens,self.num_queens)

            ################################################################################################
            ############# BREAK LOOP WHEN THERE IS NO MORE PAIRS OF ATTACKING QUEENS #######################
            # Check to see if number of attacking queens = 0, then break
            attacking_queens = current_priority - self.cost_so_far[current_state_str]
            if (current_priority != 0) and (attacking_queens == 0):
                # If difference is 0, then number of pairs of attacking queens is 0
                self.final_state = current_state
                self.end_time = time.time()
                self.duration = self.end_time - self.start_time
                # self.draw_board(self.final_state, current_priority) # debugging method
                break
            ################################################################################################
            ################################################################################################

            # If current state does not reach goal, then expand the current state
            # Getting all the row and col indices of where queens are in current state
            (row_q, col_q) = np.where(current_state == self.queen_value)
            # Sorting the col indices to ascending and using this order to sort the row indixes, this
            # would help us look at queens from the left column to the right column.
            sorted_indices = np.argsort(col_q)
            col_q_sorted = col_q[sorted_indices]
            row_q_sorted = row_q[sorted_indices]

            # Looking columns from left to right
            for c in range(self.num_queens):
                # Remove the queen in the cth column
                next_state = np.copy(current_state)
                next_state[row_q_sorted[c]][col_q_sorted[c]] = 0

                # Looking at rows from up to down within the cth column
                for r in range(self.num_queens):
                    # Place a new queen in a new row on the cth column, and with this "movement" of
                    # the queen compute the cost it took to move the queen. This if statement would
                    # only consider new movements and not the original queen position.
                    if (r != row_q_sorted[c]):
                        next_state[r][c] = self.queen_value

                        # Cost is 10 + the square of the distance of movement
                        move_cost = 10 + abs(r - row_q_sorted[c])**2
                        new_cost = self.cost_so_far[current_state_str] + move_cost

                        # Add this successsor to the priority queue or add if the cost is less than
                        # what we have seen so far.
                        next_state_str = next_state.tostring()
                        if (next_state_str not in self.cost_so_far) or (new_cost < self.cost_so_far[next_state_str]):
                            self.cost_so_far[next_state_str] = new_cost
                            # Priority is the cost of the move plus the number of pairs of attacking queens
                            # as a result of the movement.
                            self.heuristic = self.get_attack_queens(next_state)                                
                            
                            priority = new_cost + self.heuristic
                            self.frontier.put(next_state_str, priority)
                            self.came_from[next_state_str] = current_state_str

                        # Remove the queen in that state to prepare for next iteration and consider the
                        # next possible movement of the queens.
                        next_state[r][c] = 0
                    #end of for loop
                #end of for loop
            self.expansion_count += 1
            #end of while loop
                    

            
    # Class method reconstructs the set of moves by seeing where states came from.
    # Pseudocode was referenced from "Introduction to A*" from Red Blob Games
    def get_path_length(self):
        start_state_str = self.start_state.tostring()
        final_state_str = self.final_state.tostring()
        path = []
        current = final_state_str
        while current != start_state_str:
            path.append(current)
            current = self.came_from[current]
        path.append(start_state_str)
        path.reverse()
        self.path = path
        # Subtract 1 from length of path to avoid counting the start state as a "move"
        num_moves = len(path) - 1
        return num_moves
        

    
    # Class method that prints out execution time, number of moves, cost, number of expanded nodes, sequence of moves, and
    # plots the state matrix as a table using matplotlib.
    def display_solution(self):
        print "Reached a solution in", self.duration, "seconds"
        self.num_moves = self.get_path_length()
        print "Reached a solution in", self.num_moves, "moves"
        final_cost = self.cost_so_far[self.final_state.tostring()]
        print "Final cost:", final_cost
        print "Number of expanded nodes =", self.expansion_count
        print "Sequence of moves:"
        for i in range(len(self.path)):
            current_state_str = self.path[i]
            current_state = np.fromstring(current_state_str,dtype=float)
            current_state = current_state.reshape(self.num_queens,self.num_queens)
            charar_state = self.helper_charar_state(current_state)
            print "Board at state", i
            print charar_state
            print ""
        
        # Helper function plus matplotlib for better visualization
        start_charar = self.helper_charar_state(self.start_state)

        w = 8
        h = 8
        plt.figure(1, figsize=(w, h))
        tb = plt.table(cellText=start_charar, loc=(0,0), cellLoc='center')

        tc = tb.properties()['child_artists']
        for cell in tc: 
            cell.set_height(1.0/self.num_queens)
            cell.set_width(1.0/self.num_queens)

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        num_attack_queens = self.get_attack_queens(self.start_state)
        plt.title("Start State\nPairs of attacking queens = %d"%num_attack_queens)

        # Helper function plus matplotlib for better visualization
        final_charar = self.helper_charar_state(self.final_state)        
        plt.figure(2, figsize=(w, h))
        tb = plt.table(cellText=final_charar, loc=(0,0), cellLoc='center')

        tc = tb.properties()['child_artists']
        for cell in tc: 
            cell.set_height(1.0/self.num_queens)
            cell.set_width(1.0/self.num_queens)

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        num_attack_queens = self.get_attack_queens(self.final_state)
        plt.title('Final State\nPairs of attacking queens = %d, Cost = %d\nDuration = %f, Expanded Nodes = %d'%(num_attack_queens,final_cost,self.duration,self.expansion_count))
        plt.show()



    # Helper function to get a nice representation of where queens are on board. Originally queens were represented
    # with -1, but this helper function replaces the -1 with a 'Q' for better visualization.
    def helper_charar_state(self,board):
        # Getting indices of where queens are
        (row, col) = np.where(board == self.queen_value)
        charar = np.ndarray((self.num_queens,self.num_queens), dtype='S1')
        charar[:] = ' '
        for i in range(len(row)):
            charar[row[i]][col[i]] = 'Q'
        return charar
        
        
            
    # Debugging method to visual position of queens on board, where queens are denoted by -1.0    
    def draw_board(self, board, priority):
        w = 8
        h = 8
        plt.figure(1, figsize=(w, h))
        #plt.figure(figsize=(w, h))
        tb = plt.table(cellText=board, loc=(0,0), cellLoc='center')
        tc = tb.properties()['child_artists']
        for cell in tc: 
            cell.set_height(1.0/self.num_queens)
            cell.set_width(1.0/self.num_queens)
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        num_attack_queens = self.get_attack_queens(board)
        plt.title('Pairs of attacking queens = %d\nPriority = %d\nIter = %d'%(num_attack_queens,priority,self.expansion_count))
        plt.show()


        
# Priority Queue class utilizing a heap structure.
# Referenced from "Implementation of A*" from Red Blob Games
# Altered so that popping from heap would also return priority, which was used to check to see
# if the number of pairs of attacking queens is zero or not.
class PriorityQueue:
    def __init__(self):
        self.elements = []
    def empty(self):
        return len(self.elements) == 0
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    def get(self):
        #return heapq.heappop(self.elements)[1]
        return heapq.heappop(self.elements)

    
    
# Main method creates instance of an NQueenAStar class and passing N as an argument from the command line
if __name__ == '__main__':
    for arg in sys.argv[1:]:
        print "Command-line argument =", arg

    N = int(arg)
    print "Solving N Queens Problem using A* and N =", N
    game = NQueensAStar(N)

    # Running A* search
    game.a_star_search()
    # Printing and plotting solution once a final state was reached
    game.display_solution()
    print "End of N queens A* search for N =", N, "\n"
    
            
