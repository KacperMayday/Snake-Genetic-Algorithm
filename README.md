# Snake Genetic Algorithm
## About project
Main goal of this project is to show simple implementation of genetic algorithm used with neural networks. 
Required modules:
* [PyGame](https://www.pygame.org/)
* [PyTorch](https://pytorch.org/)
* [Numpy](https://numpy.org/)

## How it works

## How to run
1. Make sure that you acquired all needed modules (check requirements.txt).
2. Check config.py and adjust constants for your need (show or training mode). By default constants are ready for training. See configurations below for further explanation.
3. Run preparation.py and confirm.
4. Run main.py.
5. When training is done, adjust config.py for show mode.
6. Run main.py and see the results :)
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
SIZE = 2  
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
SIZE = 2 * 12  
POPULATION_SIZE = 100  
PARENTS_SIZE = 10  
IDLE_TIME = 2000 * 3  
MUTATION_RATE = 50  
MUTATION_FREQUENCY = 25  
CROSSING_PROBABILITY = 10  
EPOCHS = 1  
WIN_MAX = 75  
WIN_MEAN = 50  
