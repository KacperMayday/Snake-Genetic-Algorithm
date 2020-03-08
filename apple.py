import config as cfg
from random import randint
import pygame


class Apple:
    """
        A class used to represent an apple which snake chases.

        Attributes
        ----------
        size : int
            apple's size in pixels
        color : tuple
            color of the apple in RGB format

        Methods
        -------
        update(screen)
            updates apple's position on the screen
    """

    size = cfg.apple_size
    color = (250, 50, 5)

    def __init__(self):
        self.x = randint(4 * cfg.velocity, cfg.SCREENWIDTH - self.size - 4 * cfg.velocity)
        self.y = randint(4 * cfg.velocity, cfg.SCREENHEIGHT - self.size - 4 * cfg.velocity)
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

    def update(self, screen):
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        pygame.draw.rect(screen, self.color, self.rect)
