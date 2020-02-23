import pygame
from config import screen, snake_size, velocity
from snake import positions


class Sensor:
    def __init__(self, position):
        self.color = (0, 0, 0)
        self.xchange = 0
        self.ychange = 0
        distance = velocity * 2
        if position == 'up' or position == 'down':
            if position == 'up':
                self.ychange = -distance
            else:
                self.ychange = distance + snake_size
            self.width = snake_size
            self.height = 1
        else:
            if position == 'left':
                self.xchange = -distance
            else:
                self.xchange = distance + snake_size
            self.width = 1
            self.height = snake_size

        self.x = positions[-1][0]
        self.y = positions[-1][1]
        self.rect = None

    def update(self):
        self.x = positions[-1][0]
        self.y = positions[-1][1]
        self.x += self.xchange
        self.y += self.ychange
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.color, self.rect)
