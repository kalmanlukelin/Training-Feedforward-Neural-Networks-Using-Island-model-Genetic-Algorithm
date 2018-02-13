#
# ev.py: An elitist (mu+mu) generational-with-overlap EA
#
#
# To run: python ev.py --input ev3_example.cfg
#         python ev.py --input my_params.cfg
#
# Basic features of ev:
#   - Supports self-adaptive mutation
#   - Uses binary tournament selection for mating pool
#   - Uses elitist truncation selection for survivors
#

import optparse
import sys
import yaml
import math
from random import Random
from Population import *
from Individual import *
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time


#EV3 Config class 
class EV_Config:
    """
    EV configuration class
    """
    # class variables
    sectionName='DGANN'
    options={'subPopulationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'crossoverFraction': (float,True),
             'minLimit': (float,True),
             'maxLimit': (float,True),
             'numberOfIslands': (int,True),
             'frequencyOfEpochs': (int,True),
             'numberOfEpochs': (int,True),
             'fractionOfIndiviudalToMigrate': (int,True)}
     
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
    target=[[0, 1, 1, 1, 1, 0, 0]]
    input=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
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

#ANN initialization
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
ANN.input=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]]
ANN.length=len(ANN.target[0])

#Print some useful stats to screen
def printStats(pop,island):
    print('Island:',island)
    print('fitness/topology')
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
        print(ind.fit," ",ind.topologyState)
        #print(str(ind.fit))
        
    print('Min fitness',minval)
    print('Sigma',sigma)
    print('Avg fitness',avgval/len(pop))
    print('')

def calculateAvgFitness(pop):
    avgval=0
    for ind in pop:
        avgval+=ind.fit
    return avgval/len(pop)

def migration(populations, numberOfIslands, fracOfindMigrate):
    for i in range(numberOfIslands):
        for j in range(numberOfIslands):
            if i==j: 
                j+=1
            else:
                populations[i][0].combineMigrate(populations[j][0][:fracOfindMigrate])

def bestIndividual(Populations):
    islands=len(Populations)
    minval=Populations[0][0][0].fit
    bestInd=Populations[0][0][0]
    for i in range(1, islands):
        if minval > Populations[i][0][0].fit:
            minval=Populations[i][0][0].fit
            bestInd=Populations[i][0][0]
    return bestInd

#EV3:
#            
def ev(cfg):
    
    #start random number generators
    uniprng=Random()
    uniprng.seed(cfg[1])
    normprng=Random()
    normprng.seed(cfg[1]+101)
   
    #set static params on classes
    # (probably not the most elegant approach, but let's keep things simple...)
    #Individual.nLength=layer1neurons*layer1inputs+layer2neurons*layer2inputs+layer1inputs*layer2neurons
    Individual.nLength=ANN.layer1.neurons*ANN.layer1.inputsPerNeuron+ANN.layer2.neurons*ANN.layer2.inputsPerNeuron+ANN.layer1.inputsPerNeuron*ANN.layer2.neurons

    Individual.learningRate=1.0/math.sqrt(Individual.nLength)
    Individual.uniprng=uniprng
    Individual.normprng=normprng
    Individual.fitFunc=ANN.fitnessFunc
    Individual.minLimit=cfg[0].minLimit
    Individual.maxLimit=cfg[0].maxLimit
    Individual.layer1Node=ANN.layer1.neurons
    Individual.layer2Node=ANN.layer2.neurons
    Individual.inputs=ANN.layer1.inputsPerNeuron
    Population.uniprng=uniprng
    Population.crossoverFraction=cfg[0].crossoverFraction
    
    #record avgfitness in each generation
    avgfitness=[]

    if(cfg[2]):
        #create initial Population
        population=Population(cfg[0].subPopulationSize)
    else:
        #evlove population in each island
        population=cfg[3]

    avgfitness.append(calculateAvgFitness(population))
    
    if(cfg[4]):
        #final stage evolution for each islands
        Gen=cfg[4]
    else:
        #epochs
        Gen=cfg[0].frequencyOfEpochs

    #evolution main loop
    for i in range(Gen):
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
        population.truncateSelect(cfg[0].subPopulationSize)    
    
        
        #print population stats    
        #printStats(population,i+1)
        avgfitness.append(calculateAvgFitness(population))

    return population, avgfitness

def parallelEV(cfg):
    #parallel EV
    p=Pool(processes=4)
        
    #each island config
    populationCfgs=[]
    for i in range(cfg.numberOfIslands):
        populationCfgs.append((cfg, cfg.randomSeed+67*i, True, None, None))

    #each island evolve in first epoch    
    Populations=p.map(ev, populationCfgs)
    
    #record average fitness per island
    Avgfit=[]
    for i in range(cfg.numberOfIslands):
        Avgfit.append(Populations[i][1])

    #migration and evolution
    for i in range(cfg.numberOfEpochs-1):
        #migration between islands
        migration(Populations, cfg.numberOfIslands, cfg.fractionOfIndiviudalToMigrate)
        
        #each island evolve itself
        populationCfgs=[]
        for i in range(cfg.numberOfIslands):
            populationCfgs.append((cfg, cfg.randomSeed+67*i, False, Populations[i][0], None))
            
        Populations=p.map(ev, populationCfgs)
            
        #record average
        for i in range(cfg.numberOfIslands):
            Avgfit[i].extend(Populations[i][1])
        
    #each island evolve itself in the final stage
    populationCfgs=[]
    for i in range(cfg.numberOfIslands):
        populationCfgs.append((cfg, cfg.randomSeed+67*i, False, Populations[i][0], cfg.generationCount-cfg.frequencyOfEpochs*(cfg.numberOfEpochs)-4))
        
    Populations=p.map(ev, populationCfgs)
        
    #record average
    for i in range(cfg.numberOfIslands):
        Avgfit[i].extend(Populations[i][1])

    plt.xlabel('generation')
    plt.ylabel('avgfitness')
    plt.title('DGANN')
    for i in range(cfg.numberOfIslands):
        plt.plot(Avgfit[i])
    
    #
    printStats(Populations[0][0], 0)
    printStats(Populations[1][0], 1)
    printStats(Populations[2][0], 2)
    printStats(Populations[3][0], 3)
    
    bestInd = bestIndividual(Populations)
    
    print('best individual in all islands', bestInd.fit, bestInd.topologyState)
    print('')
    
    #calculation new situation
    inputs=[[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 0]]
    output=[[0, 1, 1, 1, 1, 0, 0, 0]]
    layer1out, layer2out=ANN.think(inputs, bestInd.weightState, bestInd.topologyState)
    
    print("input",inputs)
    print("Output:",layer2out)
    print("target:",output)
    
    plt.show()
    
#
# Main entry point
#
'''
def main(argv=None):
    if argv is None:
        argv = sys.argv
        
    try:
        #
        # get command-line options
        #
        parser = optparse.OptionParser()
        parser.add_option("-i", "--input", action="store", dest="inputFileName", help="input filename", default=None)
        parser.add_option("-q", "--quiet", action="store_true", dest="quietMode", help="quiet mode", default=False)
        parser.add_option("-d", "--debug", action="store_true", dest="debugMode", help="debug mode", default=False)
        (options, args) = parser.parse_args(argv)
        
        #validate options
        if options.inputFileName is None:
            raise Exception("Must specify input file name using -i or --input option.")
        
        #Get EV3 config params
        cfg=EV_Config(options.inputFileName)
        
        #print config params
        print(cfg)
        
        tstart=time.time()
        #run parallelEV
        parallelEV(cfg)
        tend=time.time()
        
        print('time elapsed',tend-tstart)
        if not options.quietMode:                    
            print('EV3 Completed!')    
    
    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)
'''

def main():
    cfg=EV_Config('DGANN.cfg')
    print(cfg)
    
    tstart=time.time()
    parallelEV(cfg)
    tend=time.time()
    
    print('time elapsed',tend-tstart)
    print('EV Completed!')

if __name__ == '__main__':
    main()
    
