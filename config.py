import pygame

SCREENWIDTH = 800
SCREENHEIGHT = 600
pygame.init()
screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT))
clock = pygame.time.Clock()
TICKRATE = 1000  # frame per sec
velocity = 5  # pixels per frame
apple_size = 5  # pixels
snake_size = 20  # pixels
positioner = snake_size//velocity
population_size = 100
parents_size = 4
idle_time = 20000  # ms
mutation_rate = 20  # in promil
mutation_frequency = 20  # in percent
crossing_probability = 20  # in percent
epoch_number = 20
