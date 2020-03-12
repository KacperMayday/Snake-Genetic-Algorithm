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
    """A class used to represent each segment of the snake.

    Snake consists of two types of segments: head and body. Only difference between
    them is that head is responsible for leading while body just follows its path.

    Attributes
    ----------
    positioner : int
        represents movement delay between each segments
    color : tuple
        color of the snake in RGB format
    size : int
        snake's size in pixels
    number : int
        unique number of the segment, head is first and starts with number 0
    xvelocity, yvelocity : int
        velocity on x and y axes
    rect : obj
        snake's segment's pygame.Rect object
    turn_counter : int
        counts turns for further scoring

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

        if self.number == 0:
            self.x = cfg.SCREENWIDTH // 2
            self.y = cfg.SCREENHEIGHT // 2
            POSITIONS.clear()
            POSITIONS.append((self.x, self.y))
            self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        else:
            try:
                self.x = POSITIONS[-self.number * self.positioner - 1][0]
                self.y = POSITIONS[-self.number * self.positioner - 1][1]
                self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
            except IndexError:
                self.x = -self.size
                self.y = -self.size
                self.rect = pygame.Rect(self.x, self.y, self.size, self.size)  # temporary rect

        self.xvelocity = 0
        self.yvelocity = 0
        self.turn_counter = 0

    def move(self, inputs):
        """Updates head's coordinates and counts turns if made.

        This method changes snake's coordinates based on inputs. Change may not be done
        if it exceeds 90 degree.

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

        # DO NOT comment these two lines
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
        self.x = randint(2 * cfg.VELOCITY, cfg.SCREENWIDTH - self.size - 2 * cfg.VELOCITY)
        self.y = randint(2 * cfg.VELOCITY, cfg.SCREENHEIGHT - self.size - 2 * cfg.VELOCITY)
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
    update(screen)
        gather last coordinates of head from global POSITIONS array and adjust the value based on which sensor
        it is, then updates its position on the screen
    """

    def __init__(self, position):
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

        self.x = POSITIONS[-1][0]
        self.y = POSITIONS[-1][1]
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)

    def update(self, screen):
        """Updates sensor's position on the screen.
        
        Parameter
        ---------
        screen : obj
            screen object where sensor is updated
        """

        self.x = POSITIONS[-1][0]  # gather last head's positions
        self.y = POSITIONS[-1][1]
        self.x += self.xchange  # adjust gathered position for specific sensor
        self.y += self.ychange
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, self.color, self.rect)
