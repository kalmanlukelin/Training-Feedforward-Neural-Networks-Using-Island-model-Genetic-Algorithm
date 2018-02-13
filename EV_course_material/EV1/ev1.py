#
# ev1.py: The simplest EA ever!
#
# To run: python ev1.py --input ev1_example.cfg
#         python ev1.py --input my_params.cfg
#
# Note: EV1 is fairly naive and has many fundamental limitations,
#           however, even though it's simple, it works!
#

import optparse
import sys
import yaml
import math
from random import Random


#EV1 Config class 
class EV1_Config:
    """
    EV1 configuration class
    """
    # class variables
    sectionName='EV1'
    options={'populationSize': (int,True),
             'generationCount': (int,True),
             'randomSeed': (int,True),
             'minLimit': (float,True),
             'maxLimit': (float,True),
             'mutationProb': (float,True),
             'mutationStddev': (float,True)}
     
    #constructor
    def __init__(self, inFileName):
        #read YAML config and get EC_Engine section
        infile=open(inFileName,'r')
        ymlcfg=yaml.load(infile)
        infile.close()
        eccfg=ymlcfg.get(self.sectionName,None)
        if eccfg is None: raise Exception('Missing EV1 section in cfg file')
         
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
         

#Simple 1-D fitness function example
#        
def fitnessFunc(x):
    return 50.0 - x*x


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
    for p in pop:
        avgval+=p.fit
        if p.fit > maxval: maxval=p.fit
        print(str(p.x)+'\t'+str(p.fit))
    
    print('Max fitness',maxval)
    print('Avg fitness',avgval/len(pop))
    print('')


#A trivial Individual class
class Individual:
    def __init__(self,x=0,fit=0):
        self.x=x
        self.fit=fit


#EV1: The simplest EA ever!
#            
def ev1(cfg):
    # start random number generator
    prng=Random()
    prng.seed(cfg.randomSeed)
    
    #random initialization of population
    population=[]
    for i in range(cfg.populationSize):
        x=prng.uniform(cfg.minLimit,cfg.maxLimit)
        ind=Individual(x,fitnessFunc(x))
        population.append(ind)
        
    #print stats    
    printStats(population,0)

    #evolution main loop
    for i in range(cfg.generationCount):
        #randomly select two parents
        parents=prng.sample(population,2)

        #recombine using simple average
        childx=(parents[0].x+parents[1].x)/2
        
        #random mutation using normal distribution
        if prng.random() <= cfg.mutationProb:
            childx=prng.normalvariate(childx,cfg.mutationStddev)
            
        #survivor selection: replace worst
        child=Individual(childx,fitnessFunc(childx))
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
        
        #Get EV1 config params
        cfg=EV1_Config(options.inputFileName)
        
        #print config params
        print(cfg)
                    
        #run EV1
        ev1(cfg)
        
        if not options.quietMode:                    
            print('EV1 Completed!')    
    
    except Exception as info:
        if 'options' in vars() and options.debugMode:
            from traceback import print_exc
            print_exc()
        else:
            print(info)
    

if __name__ == '__main__':
    main()
    
