import pygame
import config as cfg
from snake import positions


class Sensor:
    """
        A class used to represents a sensor which detects collision. They are used to produce inputs for Brain class.

        Attributes
        ----------
        color : tuple
            sensor's color in RGB format

        Methods
        -------
        update(screen)
            gather last coordinates of head from global positions array and adjust the value based on which sensor
            it is, then updates position on the screen
    """

    color = (0, 0, 0)

    def __init__(self, position):
        self.xchange = 0
        self.ychange = 0
        self.direction = position
        distance = cfg.velocity
        if position == 'up' or position == 'down':
            if position == 'up':
                self.ychange = -distance
                self.color = (255, 0, 0)
            else:
                self.ychange = cfg.snake_size
                self.color = (255, 0, 0)
            self.width = cfg.snake_size
            self.height = distance
        else:
            if position == 'left':
                self.xchange = -distance
                self.color = (0, 0, 255)
            else:
                self.xchange = cfg.snake_size
                self.color = (0, 0, 255)
            self.width = distance
            self.height = cfg.snake_size

        self.x = positions[-1][0]
        self.y = positions[-1][1]
        self.rect = None

    def update(self, screen):
        self.x = positions[-1][0]
        self.y = positions[-1][1]
        self.x += self.xchange
        self.y += self.ychange
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.color, self.rect)
