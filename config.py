"""Configuration file containing global variables and hyperparameters.

Parameters
----------
SCREENWIDTH, SCREENHEIGHT : int
    Screen resolution parameters. For training purpose I recommend setting as small values
    as possible to make the proccess faster.
TICKRATE: int
    Indicates maximum frames per second. For training set to very high value, otherwise 30-60 is recommended.
VELOCITY : int
    Number of pixels snake moves each frame. Must be a divisor of snake_size.
SIZE : int
    Size in pixels of apple and every snake's segment. Size must be divisible by velocity.
POPULATION_SIZE : int
    Size of population in each generation. Used in genetic algorithm.
PARENTS_SIZE : int
    Number of best parents chosen from each population. Must be even and a divisor of population_size.
IDLE_TIME : int
    Time in millisecond for each apple catch. Prevents snakes looping to infinity.
MUTATION_RATE : int
    Maximum per mile change during mutation.
MUTATION_FREQUENCY : int
    Mutation frequency per cent.
CROSSING_PROBABILITY : int
    Probability of exchanging each chromosome with second parent during crossing_over stage.
EPOCHS : int
    Number of epochs during training.
WIN_MAX, WIN_MEAN : int
    Winning conditions, program stops upon reaching them.
"""

SCREENWIDTH = 50
SCREENHEIGHT = 50
TICKRATE = 100000
VELOCITY = 1
SIZE = 2
POPULATION_SIZE = 1000
PARENTS_SIZE = 10
IDLE_TIME = 2500
MUTATION_RATE = 15
MUTATION_FREQUENCY = 30
CROSSING_PROBABILITY = 10
EPOCHS = 100
WIN_MAX = 75
WIN_MEAN = 40
