import argparse

### MAIN:

# Parser:
parser = argparse.ArgumentParser(description='''CS 534 Assignment 3.''')
parser.add_argument('--rGoal',dest='rGoal',nargs=1, type=int, default=5 help='''
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