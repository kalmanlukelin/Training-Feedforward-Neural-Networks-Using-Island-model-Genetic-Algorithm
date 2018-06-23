# Evolutionary Computing Project 

This project is to opitmize topology and weights of artificial neural network (ANN) using genetic algorithm(GA) with Python.

## Problem formulation

The ANN has to compute the relationship between Input and Output. After the ANN is trained, I'll give the ANN a new situation and the ANN has to solve the problem.
![Problem](https://github.com/LukeLinn/EV_project/blob/master/Figure/problem.png)

## Using genetic algorithm to solve the problem

There are 19 weights in this topology. Besides weights, there are 19 links to be optimized. 1 means connected and 0 means not. The below is the individual in GA.
![Individual](https://github.com/LukeLinn/EV_project/blob/master/Figure/individual.png)

The below are the figure for average fitness of all indivials to generation and the optimized topology:
![GA](https://github.com/LukeLinn/EV_project/blob/master/Figure/GANN.png)

## Island model GA - Improve GA with Parallel Programming

I implement multiprocess to handle four population to add variety of GA. The below are the Figure for average fitness of 4 islands to generation and the optimized topology by island model GA.
![Island_model_GA](https://github.com/LukeLinn/EV_project/blob/master/Figure/DGANN.png)

## Python modules used

* yaml
* numpy
* matplotlib

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

## Reference

* Topology design of feedforward neural networks by genetic algorithms, SlawomirW. Stepniewski and Andy J. Keane
* [How to build a simple neural network in 9 lines of Python code](https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1)