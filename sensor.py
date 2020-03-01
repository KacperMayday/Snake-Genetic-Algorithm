import pygame
from config import snake_size, velocity
from snake import positions


class Sensor:
    def __init__(self, position):
        self.color = (0, 0, 0)
        self.xchange = 0
        self.ychange = 0
        self.direction = position
        distance = velocity
        if position == 'up' or position == 'down':
            if position == 'up':
                self.ychange = -distance
                self.color = (255, 0, 0)
            else:
                self.ychange = snake_size
                self.color = (255, 0, 0)
            self.width = snake_size
            self.height = distance
        else:
            if position == 'left':
                self.xchange = -distance
                self.color = (0, 0, 255)
            else:
                self.xchange = snake_size
                self.color = (0, 0, 255)
            self.width = distance
            self.height = snake_size

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
