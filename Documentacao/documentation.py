#This file contains all the documentation about the functions implemented to solve the cartpole problem.
# Its based on the PEP 257 documentation for Python Docstrings

def truncate(f, n):
'''returns the truncated value by n decimals
param1 : the value to be truncated
param2 : the number of decimals
'''

def initialize():
'''Iterates through the carPosition, velocity, angle, and angveloc to build the states.
Assign values to right and left and writes the qtable to a file.
'''

def validState(observation):
''' Receives an observation and verify if it's already on the table.
Returns the observation in the Qtable form. (separated by pipes)
'''

def getAction(qTable, state):
'''Returns the action (left or right) with the highest value
param1 : the qTable
param2 : the actual state
'''

def updateTable(qTable, observation, reward):
'''Calculates the Qvalue based on the actual observation
Returns the best action (left or right)
param1 : the actual qTable
param2 : the observation
param3 : the value of the reward
'''
