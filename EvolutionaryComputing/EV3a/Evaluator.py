
import math

#1-D lattice total energy function evaluator class
#
class Particles1D:
    selfEnergy=None
    interactionEnergy=None
        
    @classmethod  
    def fitnessFunc(cls,state):
        totalEnergy=0
        for i in range(len(state)):
            #self energy
            totalEnergy+=cls.selfEnergy[state[i]]
            #interaction energy
            if i == 0: totalEnergy+=cls.interactionEnergy[state[i]][state[i+1]]
            elif i == len(state)-1: totalEnergy+=cls.interactionEnergy[state[i-1]][state[i]]
            else: totalEnergy+=cls.interactionEnergy[state[i-1]][state[i]] + cls.interactionEnergy[state[i]][state[i+1]]
            
        return -totalEnergy


#Multi-dimensional Rastrigrin function evaluator class
#
class Rastrigrin:
    nVars=None
    A=None
        
    @classmethod  
    def fitnessFunc(cls,state):
        fitness=cls.A*cls.nVars
        
        for i in range(cls.nVars):
            fitness+=state[i]*state[i] - cls.A*math.cos(2.0*math.pi*state[i])
            
        return -fitness
