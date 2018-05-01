'''
Universidade Federal da Fronteira Sul  - Inteligencia Artificial
Authors: Fernanda Bonetti e Matheus Henrique Trichez

A documentacao das funcoes e demais caracteristicas do trabalho podem ser encontrados em:
/Documentacao/documentation.py
/Documentacao/OpenAI Cartpole.pdf

'''
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
import math
import gym

def truncate(f, n):
    return math.floor(f * 10 ** n) / 10 ** n

#initialize the Qtable 
def initialize():
    print("Building Q-table...\n  It might take some time, it is {} states".format(len(velocity)*len(angle)*len(angveloc)*len(carPosition)))
    for h in carPosition:
        for i in velocity:
            for j in angle:
                for k in angveloc:
                    if h == -0.0:
                        h = 0.0
                    if i == -0.0:
                        i = 0.0
                    if j == -0.0:
                        j = 0.0
                    if k == -0.0:
                        k = 0.0
                    stateStr = str(truncate(h, 0))+'|'+str(truncate(i,0))+'|'+str(truncate(j, 1))+'|'+str(truncate(k, 1))
                    qTable[stateStr] = (1,1)
    with open("qFile.csv", "w") as myFile:
        pickle.dump(qTable, myFile)

#Validate the observation received
def validState(observation):
    discPos = np.round(float(observation[0]),0)
    discVel = np.round(float(observation[1]),0) #get 1 decimals
    discAngle = np.round(float(observation[2]), 1) #get 2 decimals
    discAngleVel = np.round(float(observation[3]),1) #get 1 decimals

    if discPos == -0.0:
        discPos = 0.0
    if discVel == -0.0:
        discVel = 0.0
    if discAngle == -0.0:
        discAngle = 0.0
    if discAngleVel == -0.0:
        discAngleVel = 0.0

    newState = str(discPos)+'|'+ str(discVel)+ '|'+ str(discAngle)+'|'+str(discAngleVel)
    if newState not in qTable:
        qTable[newState] = (1,1)
    return newState

def getAction(qTable, state):
    if qTable[state][0] > qTable[state][1]:
        bestAction = 0
    else:
        bestAction = 1

    return bestAction

#Calculate the Qvalues and returns the bestAction (right or left)
def updateTable(qTable, observation, reward):
    state = validState(observation)
    oldValue = max(qTable[state][0], qTable[state][1])
    qTable[state] = (reward + discountRate * qTable[state][0], reward + discountRate * qTable[state][1])
    if qTable[state][0] > qTable[state][1]:
        bestAction = 0
        qTable[state] = ((1 - learningRate) * oldValue + learningRate * (reward + discountRate * qTable[state][bestAction] - oldValue), qTable[state][1])
    else:
        bestAction = 1
        qTable[state] = (qTable[state][0], (1 - learningRate) * oldValue + learningRate * (reward + discountRate * qTable[state][bestAction] - oldValue))

    return bestAction


env = gym.make('CartPole-v0')
env.reset()
episodes = 0
carPosition = np.arange(-2, 3, 1)
velocity = np.arange(-2, 3, 1)
angle = np.arange(-2, 3, 0.1)
angveloc = np.arange(-2, 3, 0.1)

qTable = {}
#discount == 1 means that you do not favor recent actions over later actions
# so when discountRate is closest to 0 you have a really short vision
discountRate = 1
learningRate = 0.05
temps = []
tempsAll = []
tempsEveryone = []
oldState = ""
nextState = ""
maxAlive = 0

try:
    with open('qFile.csv', 'r') as handle:
        qTable = pickle.load(handle)
        print("Loading Q-table...")
except:
    initialize()

while episodes < 100:
    observation = env.reset()

    for t in range(200):
        env.render()
        oldState = validState(observation) # transform observation in a valid state
        qTable[oldState] = (1 + discountRate * qTable[oldState][0], 1 + discountRate * qTable[oldState][1])
        action = getAction(qTable, oldState)
        # in here we are on the state 's',
        observation, reward, done, info = env.step(action)
        newState = validState(observation)  # getting 's+1'
        bestFutureAction =  getAction(qTable, newState) #best action of the new state 's+1'

        #here we update Q(s,a)
        if action == 0:
            qTable[oldState] = (((1 - learningRate) * qTable[oldState][action]) + learningRate * (reward + discountRate *  qTable[newState][bestFutureAction]), qTable[oldState][1])
        else:
            qTable[oldState] = (qTable[oldState][0], (1 - learningRate) * qTable[oldState][action] + learningRate * ( reward + discountRate * qTable[newState][bestFutureAction] ))

        if done:
            if (t + 1) == 200:
                maxAlive = maxAlive + 1
            temps.append(t+1)
            tempsAll.append(t+1)
            print("Episode {} finished after {} timesteps, number of 200's: {}".format(episodes, t+1, maxAlive))
            break

    episodes = episodes + 1
    if episodes % 100 == 0:
        print("Episode: {}".format(episodes))
        print("Mean: {} ".format(np.mean(temps)))
        print("stDev: {} \n".format(np.std(temps)))
        temps[:] = []


print("\n[ALL]\nMean: {} ".format(np.mean(tempsAll)))
print("stDev: {} ".format(np.std(tempsAll)))

with open("qFile.csv", "w") as myFile:
    pickle.dump(qTable, myFile)
