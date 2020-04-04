# Snake Genetic Algorithm
## About the project
Main goal of this project is to show simple implementation of genetic algorithm used with neural networks.  
Required modules:
* [PyGame](https://www.pygame.org/)
* [PyTorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)

## How it works
### The game
The game is written in Python 3 and uses PyGame module for displaying graphics. All mechanics comes from popular old Snake game, where player controls snake's movement and the goal is to catch as many apples as possible without running into snake's tail or screen borders. In my project, snake is controlled by neural network which is trained by genetic algorithm.  
### Neural Network and Genetic Algorithm using PyTorch
The Network consists two layers: input and output layer. Input layer has 8 nodes, while output has 4 nodes. The Network is fully connected, which means that every node from input layer is connected to every node in output layer. Inputs and outputs are one hot encoded.
#### Inputs
There are 8 inputs in total, divided in two types. First four tells in which direction is apple in relation to snake's head.
Next four tells if and from which direction tail or screen border is present. Both types checks four main directions: up, right, down and left.
#### Outputs
There are 4 outputs. Outputs tell where snake should move (up, right, down or left). Output array consists of three 0 and one 1. Position of 1 determines the direction.
#### Genetic Algorithm
When whole population is scored, best individuals are paired for breeding which produces offspring based on parent's network parameters. Breeding consists of two stages: crossing-over and mutation.
During crossing-over, first parent's each network parameter from each layer (weight or bias) has a chance to be replaced by a parameter from the second parent. After crossing-over, child's network is exposed to mutation, which randomly alters its parameters. Then, children are saved to population and next generation is prepared for scoring.
## How to run
1. Make sure that you acquired all needed modules (check requirements.txt).
2. Check config.py and adjust constants for your need (show or training mode). By default constants are ready for training. See configurations below for further explanation.
3. Run main.py -r and confirm.
5. When training is done, adjust config.py for show mode.
6. Run main.py -s [number] to display population [number] times and see the results :)
7. Play with the constants and see different outcomes after training.

## Configurations
Main differents between training and display modes are: 
* resolution parameters - for training use as small resolution as possible, because the game runs faster on smaller screens
* tickrate -  determines maximum number of frames per second, so it should be a huge number
* velocity - adjust to match screen scale
* size - adjust to match screen scale
* idle time - determines how much time in ms snake has to reach next apple. Set small on smaller screens.
* epochs - number of epochs for training. If set to 1, game is set to show mode. 

Below you can find example configurations.
#### For training:
SCREENWIDTH = 50  
SCREENHEIGHT = 50  
TICKRATE = 10000  
VELOCITY = 1  
SIZE = VELOCITY  
POPULATION_SIZE = 100  
PARENTS_SIZE = 10  
IDLE_TIME = 2000  
MUTATION_RATE = 50  
MUTATION_FREQUENCY = 25  
CROSSING_PROBABILITY = 10  
EPOCHS = 100  
WIN_MAX = 75  
WIN_MEAN = 50  
#### For display:
SCREENWIDTH = 50 * 12  
SCREENHEIGHT = 50 * 12  
TICKRATE = 30  
VELOCITY = 1 * 12  
SIZE = VELOCITY   
POPULATION_SIZE = 100  
PARENTS_SIZE = 10  
IDLE_TIME = 2000 * 3  
MUTATION_RATE = 50  
MUTATION_FREQUENCY = 25  
CROSSING_PROBABILITY = 10  
EPOCHS = 1  
WIN_MAX = 75  
WIN_MEAN = 50  
