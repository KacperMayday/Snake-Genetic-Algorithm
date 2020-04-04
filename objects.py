"""Module containing objects used in the game."""

from random import randint

import pygame

import config as cfg


class Snake:
    """A class used to represent each segment of the snake.

    Snake consists of two types of segments: head and body. Only difference between
    them is that head is responsible for leading while body just follows its path.

    Attributes
    ----------
    color : tuple
        color of the snake in RGB format
    size : int
        snake's size in pixels
    x, y : int
        object's coordinates
    xvelocity, yvelocity : int
        velocity on x and y axes
    rect : obj
        snake's segment's pygame.Rect object
    turn_counter : int
        counts turns for further scoring

    Methods
    -------
    move(inputs, screen)
        encodes inputs array to direction in which snake's head should move and
        updates it on the given screen
    update(screen)
        updates segment position on the screen
    set_coordinates(x, y)
        changes segment's coordinates
    """

    def __init__(self,
                 x=randint(0, cfg.SCREENWIDTH // cfg.SIZE) * cfg.SIZE,
                 y=randint(0, cfg.SCREENHEIGHT // cfg.SIZE) * cfg.SIZE):
        self.color = (0, 0, 0)
        self.size = cfg.SIZE
        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)

        self.xvelocity = 0
        self.yvelocity = 0
        self.turn_counter = 0

    def move(self, inputs, screen):
        """Updates head's coordinates and counts turns if made.

        This method changes snake's coordinates based on inputs. Change may not be done
        if it is not 90 degree.

        Note
        ----
        There are two blocks of code commented. They allow extra functionality if uncommented.
        First allows player control the snake with arrows. Second enables wall teleports so
        when snake cross top border it will appear on the bottom. To enable first functionality
        remember to comment whole code above first block. To enable second one remember to
        comment corresponding game condition in game.py, check() method.

        Parameter
        ---------
        inputs : list
            one-hot inputs array indicating next move direction
        screen : obj
            screen on which object is displayed
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

        if xtemp != self.xvelocity and ytemp != self.yvelocity:  # check if snake changed direction
            self.turn_counter += 1

        self.x += self.xvelocity
        self.y += self.yvelocity

        # Uncomment this block below to have control over snake's head:
        # pressed = pygame.key.get_pressed()
        # if pressed[pygame.K_UP]:
        #     self.yvelocity = -cfg.VELOCITY
        #     self.xvelocity = 0
        # elif pressed[pygame.K_DOWN]:
        #     self.yvelocity = cfg.VELOCITY
        #     self.xvelocity = 0
        # elif pressed[pygame.K_RIGHT]:
        #     self.xvelocity = cfg.VELOCITY
        #     self.yvelocity = 0
        # elif pressed[pygame.K_LEFT]:
        #     self.xvelocity = -cfg.VELOCITY
        #     self.yvelocity = 0
        # self.x += self.xvelocity
        # self.y += self.yvelocity

        # Uncomment this block to enable wall teleports
        # Remember to comment necessary game condition!
        # if self.x > cfg.SCREENWIDTH + self.size:
        #     self.x += -cfg.SCREENWIDTH - 2 * self.size
        # elif self.x < -self.size:
        #     self.x += cfg.SCREENWIDTH + self.size
        #
        # if self.y > cfg.SCREENHEIGHT + self.size:
        #     self.y += -cfg.SCREENHEIGHT - 2 * self.size
        # elif self.y < -self.size:
        #     self.y += cfg.SCREENHEIGHT + self.size

        # DO NOT comment this line
        self.update(screen)

    def update(self, screen):
        """Updates snake's position on the screen.

        Parameter
        ---------
        screen : obj
            screen object where snake is updated
        """

        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        pygame.draw.rect(screen, self.color, self.rect)

    def set_coordinates(self, x, y):
        """Set coordinates corresponding to body segment.

        Parameter
        ---------
        x, y : int
            object's coordinates
        """
        self.x = x
        self.y = y


class Apple:
    """A class used to represent an apple which snake chases.

    Note
    ----
    Note that apple's x and y is limited to make sure snake is able to catch it
    without going beyond screen borders.

    Attributes
    ----------
    size : int
        apple's size in pixels
    color : tuple
        color of the apple in RGB format
    x, y : int
        apple's coordinates
    rect : obj
        apple's pygame.Rect object

    Methods
    -------
    update(screen)
        updates apple's position on the screen
    """

    def __init__(self):
        self.size = cfg.SIZE
        self.color = (250, 50, 5)
        self.x = randint(0, (cfg.SCREENWIDTH - self.size) // self.size) * self.size
        self.y = randint(0, (cfg.SCREENHEIGHT - self.size) // self.size) * self.size
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
    position : str
        sensor position in relation to snake's head; up, right, down or left
    color : tuple
        sensor's color in RGB format
    xchange, ychange : int
        change of x and y in relation to snake's head
    width, height : int
        width and height of the sensor
    x, y : int
        sensor's coordinates
    rect : obj
        sensor's pygame.Rect object

    Methods
    -------
    update(screen, x, y)
        updates sensor's position in relation to head's coordinates
    """

    def __init__(self, position, x, y):
        self.color = (0, 0, 0)
        self.xchange = 0
        self.ychange = 0
        distance = cfg.VELOCITY  # distance between sensor and snake's head
        if position == 'up' or position == 'down':  # determine sensor position
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

        self.x = x
        self.y = y
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, screen, x, y):
        """Updates sensor's position on the screen.
        
        Parameter
        ---------
        screen : obj
            screen object where sensor is updated
        x, y : int
            head's coordinates
        """

        self.x = x
        self.y = y
        self.x += self.xchange  # adjust gathered position for specific sensor
        self.y += self.ychange
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.color, self.rect)
