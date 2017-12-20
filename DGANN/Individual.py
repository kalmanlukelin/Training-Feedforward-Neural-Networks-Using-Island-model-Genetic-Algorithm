#
# Individual.py
#
#

import math

#Base class for all individual types
#
class Individual:
    """
    Individual
    """
    minMutRate=1e-100
    maxMutRate=1
    nLength=None
    learningRate=None
    uniprng=None
    normprng=None
    fitFunc=None
    minLimit=None
    maxLimit=None
    layer1Node=None
    layer2Node=None
    inputs=None

    def __init__(self):
        
        self.weightState=[]
        for i in range(self.nLength):
            self.weightState.append(self.uniprng.uniform(self.minLimit,self.maxLimit))
        
        self.topologyState=[]
        
        for i in range(self.nLength):
            self.topologyState.append(self.uniprng.randrange(2))

        self.fit=self.__class__.fitFunc(self.weightState, self.topologyState)
        self.mutRate=self.uniprng.uniform(0.9,0.1) #use "normalized" sigma
        
    def checkSurvival(self, topology):
        input1=False
        input2=False
        input3=False
    
        tmp=self.layer1Node*self.inputs
        if(topology[tmp]): input1=True
        if(topology[tmp+1]): input2=True
        if(topology[tmp+2]): input3=True
        
        tmp+=self.inputs
        for i in range(self.layer1Node):
            if(topology[tmp+i]):
                if(topology[self.inputs*i]):
                    input1=True
                if(topology[self.inputs*i+1]):
                    input2=True
                if(topology[self.inputs*i+2]):
                    input3=True
        
        return input1 and input2 and input3
        
    def crossover(self, other):
        
        #topology crossover
        #record original topology
        originalTopologySelf=self.topologyState
        originalTopologyOther=other.topologyState
        
        #perform crossover "in-place"
        for i in range(self.nLength):
            if self.uniprng.random() < 0.5:
                tmp=self.topologyState[i]
                self.topologyState[i]=other.topologyState[i]
                other.topologyState[i]=tmp
    
        #weight crossover
        alpha=self.uniprng.random()
        
        for i in range(self.nLength):
            tmp=self.weightState[i]*alpha+other.weightState[i]*(1-alpha)
            other.weightState[i]=self.weightState[i]*(1-alpha)+other.weightState[i]*alpha
            self.weightState[i]=tmp
            
            if self.weightState[i] > self.maxLimit: self.weightState[i]=self.maxLimit
            if self.weightState[i] < self.minLimit: self.weightState[i]=self.minLimit
            if other.weightState[i] > self.maxLimit: other.weightState[i]=self.maxLimit
            if other.weightState[i] < self.minLimit: other.weightState[i]=self.minLimit
                
        self.fit=None
        other.fit=None
            
    def mutate(self):
        self.mutRate=self.mutRate*math.exp(self.learningRate*self.normprng.normalvariate(0,1))
        if self.mutRate < self.minMutRate: self.mutRate=self.minMutRate
        if self.mutRate > self.maxMutRate: self.mutRate=self.maxMutRate
        
        #topology mutation
        for i in range(self.nLength):
            if self.uniprng.random() < self.mutRate:
                self.topologyState[i]=self.uniprng.randrange(2)

        #weight mutation
        for i in range(self.nLength):
            self.weightState[i]=self.weightState[i]+(self.maxLimit-self.minLimit)*self.mutRate*self.normprng.normalvariate(0,1)
            if self.weightState[i] > self.maxLimit: self.weightState[i]=self.maxLimit
            if self.weightState[i] < self.minLimit: self.weightState[i]=self.minLimit
        
        self.fit=None

    def evaluateFitness(self):
        if self.fit == None: self.fit=self.__class__.fitFunc(self.weightState, self.topologyState)

