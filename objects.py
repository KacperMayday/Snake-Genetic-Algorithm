"""Module containing objects used in the game.

Global list POSITIONS is filled by head's coordinates every frame. There are needed
to determine position of sensors and snake's body segments. This allows snake's body
to follow exact path which head takes.
"""

from random import randint

import pygame

import config as cfg

POSITIONS = []


class Snake:
    """A class used to represent each segment of the snake as well as its head.

    Attributes
    ----------
    positioner : int
        represents movement delay between each segments
    color : tuple
        color of the snake in RGB format
    size : int
        snake's size in pixels
    number : int

    xvelocity, yvelocity : int

    rect : obj

    turn_counter : int


    Methods
    -------
    move(inputs)
        encodes inputs array to direction in which snake's head should move
        and append head's new coordinates to global POSITIONS array
    update(screen)
        updates segment position on the screen
    body_move(screen)
        gather coordinates from global POSITIONS array for the next move for each segment
    """

    def __init__(self, number):
        self.positioner = cfg.SIZE // cfg.VELOCITY
        self.color = (0, 0, 0)
        self.size = cfg.SIZE
        self.number = number
        global POSITIONS

        try:
            if self.number == 0:
                raise IndexError
            self.x = POSITIONS[-self.number * self.positioner - 1][0]
            self.y = POSITIONS[-self.number * self.positioner - 1][1]

        except IndexError:
            self.x = cfg.SCREENWIDTH // 2
            self.y = cfg.SCREENHEIGHT // 2
            POSITIONS.clear()
            POSITIONS.append((self.x, self.y))

        self.xvelocity = 0
        self.yvelocity = 0
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        self.turn_counter = 0

    def move(self, inputs):
        """

        Parameter
        ---------
        inputs : list

        """

        xtemp = self.xvelocity
        ytemp = self.yvelocity

        if inputs[0] == 1 and self.yvelocity != cfg.VELOCITY:  # pressed[pygame.K_UP]:
            self.yvelocity = -cfg.VELOCITY
            self.xvelocity = 0
        elif inputs[2] == 1 and self.yvelocity != -cfg.VELOCITY:  # pressed[pygame.K_DOWN]:
            self.yvelocity = cfg.VELOCITY
            self.xvelocity = 0
        elif inputs[1] == 1 and self.xvelocity != -cfg.VELOCITY:  # pressed[pygame.K_RIGHT]:
            self.xvelocity = cfg.VELOCITY
            self.yvelocity = 0
        elif inputs[3] == 1 and self.xvelocity != cfg.VELOCITY:  # pressed[pygame.K_LEFT]:
            self.xvelocity = -cfg.VELOCITY
            self.yvelocity = 0

        # pressed = pygame.key.get_pressed()
        # if pressed[pygame.K_UP]:
        #     self.yvelocity = -velocity
        #     self.xvelocity = 0
        # elif pressed[pygame.K_DOWN]:
        #     self.yvelocity = velocity
        #     self.xvelocity = 0
        # elif pressed[pygame.K_RIGHT]:
        #     self.xvelocity = velocity
        #     self.yvelocity = 0
        # elif pressed[pygame.K_LEFT]:
        #     self.xvelocity = -velocity
        #     self.yvelocity = 0

        if xtemp != self.xvelocity and ytemp != self.yvelocity:
            self.turn_counter += 1

        self.x += self.xvelocity
        self.y += self.yvelocity

        # if self.x > SCREENWIDTH + self.size:
        #     self.x += -SCREENWIDTH - 2 * self.size
        # elif self.x < -self.size:
        #     self.x += SCREENWIDTH + self.size
        #
        # if self.y > SCREENHEIGHT + self.size:
        #     self.y += -SCREENHEIGHT - 2 * self.size
        # elif self.y < -self.size:
        #     self.y += SCREENHEIGHT + self.size

        global POSITIONS
        POSITIONS.append((self.x, self.y))

    def update(self, screen):
        """Updates snake's position on the screen.

        Parameter
        ---------
        screen : obj
            screen object where snake is updated
        """

        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        pygame.draw.rect(screen, self.color, self.rect)

    def body_move(self):
        """Gathers coordinates corresponding to body segment.

        It may happen that POSITIONS array is not filled yet with enough coordinates
        so the segment can't acquire correct x and y. Since POSITIONS is updated quite
        fast, this problem is not significant and barely visible.
        """

        try:
            self.x = POSITIONS[-self.number*self.positioner - 1][0]
            self.y = POSITIONS[-self.number*self.positioner - 1][1]
        except IndexError:
            pass


class Apple:
    """A class used to represent an apple which snake chases.

    Attributes
    ----------
    size : int
        apple's size in pixels
    color : tuple
        color of the apple in RGB format
    x, y : int

    rect : obj


    Methods
    -------
    update(screen)
        updates apple's position on the screen
    """

    def __init__(self):
        self.size = cfg.SIZE
        self.color = (250, 50, 5)
        self.x = randint(4 * cfg.VELOCITY, cfg.SCREENWIDTH - self.size - 4 * cfg.VELOCITY)
        self.y = randint(4 * cfg.VELOCITY, cfg.SCREENHEIGHT - self.size - 4 * cfg.VELOCITY)
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

    def update(self, screen):
        """Updates apple's position on the screen.

        Parameter
        ---------
        screen : obj
            screen object where apple is updated
        """

        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        pygame.draw.rect(screen, self.color, self.rect)


class Sensor:
    """A class used to represents a sensor which detects collision. They are used
    to produce inputs for neural network.

    Attributes
    ----------
    color : tuple
        sensor's color in RGB format
    xchange, ychange : int

    direction : str

    color : tuple

    xchange, ychange : int

    width, height : int

    x, y : int

    rect : obj


    Methods
    -------
    update(screen)
        gather last coordinates of head from global POSITIONS array and adjust the value based on which sensor
        it is, then updates its position on the screen
    """

    def __init__(self, position):
        self.color = (0, 0, 0)
        self.xchange = 0
        self.ychange = 0
        self.direction = position
        distance = 2 * cfg.VELOCITY
        if position == 'up' or position == 'down':
            if position == 'up':
                self.ychange = -distance
            else:
                self.ychange = cfg.SIZE
            self.color = (255, 0, 0)
            self.width = cfg.SIZE
            self.height = distance

        else:
            if position == 'left':
                self.xchange = -distance
            else:
                self.xchange = cfg.SIZE
            self.color = (0, 0, 255)
            self.width = distance
            self.height = cfg.SIZE

        self.x = POSITIONS[-1][0]
        self.y = POSITIONS[-1][1]
        self.rect = None

    def update(self, screen):
        """Updates sensor's position on the screen.
        
        Parameter
        ---------
        screen : obj
            screen object where sensor is updated
        """

        self.x = POSITIONS[-1][0]
        self.y = POSITIONS[-1][1]
        self.x += self.xchange
        self.y += self.ychange
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.color, self.rect)
