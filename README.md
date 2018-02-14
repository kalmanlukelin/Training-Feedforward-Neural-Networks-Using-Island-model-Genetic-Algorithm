# Evolutionary Computing Project 

This project is to opitmize topology and weights of artificial neural network (ANN) using genetic algorithm(GA).

## Problem formulation

The ANN has to compute the relationship between Input and Output. After the ANN is trained, I'll give the ANN a new situation and the ANN has to solve the problem.
![Problem](https://github.com/LukeLinn/EV_project/blob/master/picture/problem.png)

## Using genetic algorithm to solve the problem

The original ANN topology I set
![Original_topology](https://github.com/LukeLinn/EV_project/blob/master/picture/original_topology.png)

The individual in GA
![Individual](https://github.com/LukeLinn/EV_project/blob/master/picture/individual.png)

There are 19 weights in this topology. Besides weights, there are 19 links to be optimized. 1 means connected and 0 means not.

Optimized topology by GA
![GA](https://github.com/LukeLinn/EV_project/blob/master/picture/GA.png)

Figure for Average fitness of all indivials in GA to generation
![GA_figure](https://github.com/LukeLinn/EV_project/blob/master/picture/GANN.png)

## Improve GA with Parallel Programming

Island model topology
![Island_model](https://github.com/LukeLinn/EV_project/blob/master/picture/island_model.png)

I implement multiprocess to improve GA. Four processes here handle four population. The numbers near each nodes are the randomseed I set.

Figure for Average fitness of 4 islands to generation
![Island_model_GA](https://github.com/LukeLinn/EV_project/blob/master/picture/island_model_GA.png)

## Reference

* Topology design of feedforward neural networks by genetic algorithms, SlawomirW. Stepniewski and Andy J. Keane
* [How to build a simple neural network in 9 lines of Python code](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1)