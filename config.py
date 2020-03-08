"""
    Configuration file containing global variables and hyperparameters

    Parameters
    ----------
    SCREENWIDTH, SCREENHEIGHT : int
        Screen resolution parameters. For training purpose I recommend setting as small values
        as possible to make the proccess faster.
    TICKRATE: int
        Indicates maximum frames per second. For training set to very high value, otherwise 30-60 is recommended.
    velocity : int
        Number of pixels snake moves each frame. Must be a divisor of snake_size.
    apple_size, snake_size : int
        Size in pixels of apple and every snake's segment. Snake size must be divisible by velocity.
    population_size : int
        Size of population in each generation. Used in genetic algorithm.
    parents_size : int
        Number of best parents chosen from each population. Must be even and a divisor of population_size.
    idle_time : int
        Time in millisecond for each apple catch. Prevents snakes looping to infinity.
    mutation_rate : int
        Maximum per mile change during mutation.
    mutation_frequency : int
        Mutation frequency per cent.
    crossing_probability : int
        Probability of exchanging each chromosome with second parent during crossing_over stage.
    epoch_number : int
        Number of epochs during training.
"""

SCREENWIDTH = 800
SCREENHEIGHT = 600
TICKRATE = 60
velocity = 5
apple_size = 10
snake_size = 20
population_size = 2
parents_size = 2
idle_time = 5000
mutation_rate = 300
mutation_frequency = 30
crossing_probability = 10
epoch_number = 1
