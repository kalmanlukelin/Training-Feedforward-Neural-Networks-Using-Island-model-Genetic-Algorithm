#
# ev2.py: ev1 with the following modifications:
#          - self-adaptive mutation
#          - stochastic arithmetic crossover
#          - restructured code for better use of OO
#
# Note: EV2 still suffers from many of the weaknesses of EV1,
#       most particularly in the parent/survivor selection processes
#
# To run: python ev2.py --input ev2_example.cfg
#         python ev2.py --input my_params.cfg
#
#

import optparse
import sys
import yaml
import math
from random import Random


#EV2 Config class 
class EV2_Config:
    """
    EV2 configuration class
    """
    # class variables
    sectionName='EV2'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'minLimit': (float,True),
             'maxLimit': (float,True)}
     
    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EV2 section
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
         

#Simple 1-D fitness function example: 1-D Rastrigrin function
#        
def fitnessFunc(x):
    return -10.0-(0.04*x)**2+10.0*math.cos(0.04*math.pi*x)


#Find index of worst individual in population
def findWorstIndex(l):
    minval=l[0].fit
    imin=0
    for i in range(len(l)):
        if l[i].fit < minval:
            minval=l[i].fit
            imin=i
    return imin


#Print some useful stats to screen
def printStats(pop,gen):
    print('Generation:',gen)
    avgval=0
    maxval=pop[0].fit 
    sigma=pop[0].sigma
    for ind in pop:
        avgval+=ind.fit
        if ind.fit > maxval:
            maxval=ind.fit
            sigma=ind.sigma
        print(str(ind.x)+'\t'+str(ind.fit)+'\t'+str(ind.sigma))

    print('Max fitness',maxval)
    print('Sigma',sigma)
    print('Avg fitness',avgval/len(pop))
    print('')


#A simple Individual class
class Individual:
    minSigma=1e-100
    maxSigma=1
    #Note, the learning rate is typically tau=A*1/sqrt(problem_size)
    # where A is a user-chosen scaling factor (optional) and problem_size
    # for real and integer vector problems is usually the vector-length.
    # In our case here, the vector length is 1, so we choose to use a learningRate=1
    learningRate=1
    minLimit=None
    maxLimit=None
    cfg=None
    prng=None
    fitFunc=None

    def __init__(self,randomInit=True):
        if randomInit:
            self.x=self.prng.uniform(self.minLimit,self.maxLimit)
            self.fit=self.__class__.fitFunc(self.x)
            self.sigma=self.prng.uniform(0.9,0.1) #use "normalized" sigma
        else:
            self.x=0
            self.fit=0
            self.sigma=self.minSigma
        
    def crossover(self, other):
        child=Individual(randomInit=False)
        alpha=self.prng.random()
        child.x=self.x*alpha+other.x*(1-alpha)
        child.sigma=self.sigma*alpha+other.sigma*(1-alpha)
        child.fit=None
        
        return child
    
    def mutate(self):
        self.sigma=self.sigma*math.exp(self.learningRate*self.prng.normalvariate(0,1))
        if self.sigma < self.minSigma: self.sigma=self.minSigma
        if self.sigma > self.maxSigma: self.sigma=self.maxSigma

        self.x=self.x+(self.maxLimit-self.minLimit)*self.sigma*self.prng.normalvariate(0,1)
    
    def evaluateFitness(self):
        self.fit=self.__class__.fitFunc(self.x)


#EV2: EV1 with self-adaptive mutation & stochastic crossover
#            
def ev2(cfg):
    #start random number generator
    prng=Random()
    prng.seed(cfg.randomSeed)

    #set Individual static params: min/maxLimit, fitnessFunc, & prng
    Individual.minLimit=cfg.minLimit
    Individual.maxLimit=cfg.maxLimit
    Individual.fitFunc=fitnessFunc
    Individual.prng=prng
      
    #random initialization of population
    population=[]
    for i in range(cfg.populationSize):
        ind=Individual()
        ind.evaluateFitness()
        population.append(ind)
        
    #print stats    
    printStats(population,0)

    #evolution main loop
    for i in range(cfg.generationCount):
        #randomly select two parents
        parents=prng.sample(population,2)

        #recombine
        child=parents[0].crossover(parents[1])
        
        #random mutation
        child.mutate()
        
        #update child's fitness value
        child.evaluateFitness()        
            
        #survivor selection: replace worst
        iworst=findWorstIndex(population)
        if child.fit > population[iworst].fit:
            population[iworst]=child
        
        #print stats    
        printStats(population,i+1)
        
        
#
# Main entry point
#
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
        
        #Get EV2 config params
        cfg=EV2_Config(options.inputFileName)
        
        #print config params
        print(cfg)
                    
        #run EV2
        ev2(cfg)
        
        if not options.quietMode:                    
            print('EV2 Completed!')    
    
    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)
    

if __name__ == '__main__':
    main()
    
