from config import SCREENHEIGHT, SCREENWIDTH, apple_size, velocity
from random import randint
import pygame


class Apple:
    def __init__(self):
        self.size = apple_size
        self.x = randint(4 * velocity, SCREENWIDTH - self.size - 4 * velocity)
        self.y = randint(4 * velocity, SCREENHEIGHT - self.size - 4 * velocity)
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        self.color = (250, 50, 5)

    def update(self, screen):
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        pygame.draw.rect(screen, self.color, self.rect)
