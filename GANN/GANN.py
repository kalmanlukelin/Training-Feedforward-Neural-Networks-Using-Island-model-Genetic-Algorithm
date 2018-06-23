import optparse
import sys
import yaml
import math
from random import Random
from Population import *
from Individual import *
import numpy as np
import matplotlib.pyplot as plt
import time

#EV3 Config class 
class EV_Config:
    """
    EV configuration class
    """
    # class variables
    sectionName='EV3'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'crossoverFraction': (float,True),
             'minLimit': (float,True),
             'maxLimit': (float,True)}
     
    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EV3 section
        infile=open(inFileName,'r')
        ymlcfg=yaml.load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing {} section in cfg file'.format(self.sectionName))
         
        #iterate over options
        for opt in self.options:
            if opt in eccfg:
                optval=eccfg[opt]
 
                #verify parameter type
                if type(optval) != self.options[opt][0]:
                    raise Exception('Parameter "{}" has wrong type'.format(opt))
                 
                #create attributes on the fly
                setattr(self,opt,optval)
            else:
                if self.options[opt][1]:
                    raise Exception('Missing mandatory parameter "{}"'.format(opt))
                else:
                    setattr(self,opt,None)
     
    #string representation for class data    
    def __str__(self):
        return str(yaml.dump(self.__dict__,default_flow_style=False))

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.neurons=number_of_neurons
        self.inputsPerNeuron=number_of_inputs_per_neuron
        
class ANN:
    target=None
    input=None
    layer1=None
    layer2=None
    length=None
    
    @classmethod
    def fitnessFunc(cls, weightstate, topologystate):
        output_from_layer1, output_from_layer2=cls.think(cls.input, weightstate, topologystate)
        cls.think(cls.input, weightstate, topologystate)
        return np.sum(np.square(cls.target-output_from_layer2))/(2*cls.length)
    
    @classmethod
    # The neural network thinks.
    def think(cls, inputs, weightstate, topologystate):
        #get layer1SynapticWeight matrix
        i=0
        layer1SynapticWeight=[]
        tmp=[]
        
        while(i < cls.layer1.inputsPerNeuron*cls.layer1.neurons):
            for j in range(cls.layer1.inputsPerNeuron):
                tmp.append(weightstate[i+j]*topologystate[i+j])
            layer1SynapticWeight.append(tmp)
            tmp=[]
            i+=cls.layer1.inputsPerNeuron
        layer1SynapticWeight=np.array(layer1SynapticWeight).T
        
        #get layer2 SynapticWeight matrix
        i=cls.layer1.inputsPerNeuron*cls.layer1.neurons
        layer2SynapticWeight=[]
        
        length=len(weightstate)
        for j in range(i, length):
            layer2SynapticWeight.append(weightstate[j]*topologystate[j])
        
        
        #calculate output for each layer
        output_from_layer1 = cls().__sigmoid(np.dot(inputs, layer1SynapticWeight))

        #append the shortcut from inputs to the second layer
        inputs2=np.append(inputs, output_from_layer1, axis=1)
        
        output_from_layer2 = cls().__sigmoid(np.dot(inputs2, layer2SynapticWeight))

        return output_from_layer1, output_from_layer2
    
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

#Print some useful stats to screen
def printStats(pop,gen):
    print('Generation:',gen)
    #print('topologyState/weightState/fitness/sigma')
    avgval=0
    minval=pop[0].fit 
    sigma=pop[0].mutRate
    for ind in pop:
        avgval+=ind.fit
        if ind.fit < minval:
            minval=ind.fit
            sigma=ind.mutRate
        #print(ind.weightState+ind.topologyState+ind.fit+ind.mutRate)
        #print(str(ind.topologyState)+str(ind.weightState)+str(ind.fit)+str(ind.mutRate))
        print(str(ind.fit)+" "+str(ind.topologyState)+" "+str(ind.weightState))
    print('Min fitness',minval)
    print('Sigma',sigma)
    print('Avg fitness',avgval/len(pop))
    print('')

def calculateAvgFitness(pop):
    avgval=0
    for ind in pop:
        avgval+=ind.fit
    return avgval/len(pop)
    
#EV3:
#            
def ev(cfg):
    
    #start random number generators
    uniprng=Random()
    uniprng.seed(cfg.randomSeed)
    normprng=Random()
    normprng.seed(cfg.randomSeed+101)
    
    #Create layer 1 
    layer1neurons=4
    layer1inputs=3
    layer1 = NeuronLayer(layer1neurons, layer1inputs)
    
    #Create layer 2
    layer2neurons=1
    layer2inputs=layer1neurons
    layer2 = NeuronLayer(layer2neurons, layer2inputs)
    
    #Set ANN
    ANN.layer1=layer1
    ANN.layer2=layer2
    ANN.target=[[0, 1, 1, 1, 1, 0, 0]]
    ANN.length=len(ANN.target[0])
    ANN.input=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
    
    #set static params on classes
    # (probably not the most elegant approach, but let's keep things simple...)
    Individual.nLength=layer1neurons*layer1inputs+layer2neurons*layer2inputs+layer1inputs*layer2neurons
    Individual.learningRate=1.0/math.sqrt(Individual.nLength)
    Individual.uniprng=uniprng
    Individual.normprng=normprng
    Individual.fitFunc=ANN.fitnessFunc
    Individual.minLimit=cfg.minLimit
    Individual.maxLimit=cfg.maxLimit
    Individual.layer1Node=layer1.neurons
    Individual.layer2Node=layer2.neurons
    Individual.inputs=layer1.inputsPerNeuron
    Population.uniprng=uniprng
    Population.crossoverFraction=cfg.crossoverFraction
    
    #record avgfitness in each generation
    avgfitness=[]
    
    #create initial Population (random initialization)
    population=Population(cfg.populationSize)
    
    #print initial pop stats    
    printStats(population,0)
    avgfitness.append(calculateAvgFitness(population))
    
    #evolution main loop
    for i in range(cfg.generationCount):
        #create initial offspring population by copying parent pop
        offspring=population.copy()
       
        #select mating pool
        offspring.conductTournament()
    
        #perform crossover
        offspring.crossover()
    
        #random mutation
        offspring.mutate()
        
        #update fitness values
        offspring.evaluateFitness()        
            
        #survivor selection: elitist truncation using parents+offspring
        population.combinePops(offspring)
        population.truncateSelect(cfg.populationSize)
        
        #print population stats    
        #printStats(population,i+1)
        avgfitness.append(calculateAvgFitness(population))
    
    printStats(population,i+1)

    #calculation new situation
    inputs=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]]
    output=[[0, 1, 1, 1, 1, 0, 0, 0]]
    layer1out, layer2out=ANN.think(inputs, population[0].weightState, population[0].topologyState)
    print("input"+str(inputs))
    print("Output:"+str(layer2out))
    print("target:"+str(output))
    
    #print fitness value in each generation 
    plt.plot(avgfitness)
    plt.ylabel('avgfitness')
    plt.xlabel('generation')
    plt.title('GANN')
    plt.show()
#
# Main entry point
#

def main():
    cfg=EV_Config('GANN.cfg')
    print(cfg)
    
    tstart=time.time() 
    ev(cfg)
    tend=time.time()
    
    print('time elapse',tend-tstart)
    print('EV Completed!')

if __name__ == '__main__':
    main()
    
