# Evolutionary Computing Project 

This project is to opitmize topology and weights of artificial neural network (ANN) using genetic algorithm(GA).

## Problem formulation

The ANN has to compute the relationship between Input and Output. After the ANN is trained, I'll give the ANN a new situation and the ANN has to solve the problem.
![Problem](https://github.com/LukeLinn/EV_project/blob/master/figure/problem.png)

## Using genetic algorithm to solve the problem

The individual in GA

![Individual](https://github.com/LukeLinn/EV_project/blob/master/figure/individual.png)

There are 19 weights in this topology. Besides weights, there are 19 links to be optimized. 1 means connected and 0 means not.

Optimized topology by GA and figure for Average fitness of all indivials in GA to generation

![GA](https://github.com/LukeLinn/EV_project/blob/master/figure/GANN.png)

## Island model GA - Improve GA with Parallel Programming

I implement multiprocess to improve GA. I implement four processes to handle four population.

Optimized topology by island model GA and figure for Average fitness of 4 islands to generation

![Island_model_GA](https://github.com/LukeLinn/EV_project/blob/master/figure/DGANN.png)

## Running the project

To see the result of GA
```
cd GANN
python GANN.py
```

To see the result of island model GA
```
cd DGANN
python DGANN.py
```

## Python modules used

* yanl
* numpy
* matplotlib

## Reference

* Topology design of feedforward neural networks by genetic algorithms, SlawomirW. Stepniewski and Andy J. Keane
* [How to build a simple neural network in 9 lines of Python code](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1)