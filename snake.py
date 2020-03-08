import pygame
import config as cfg
positions = []


class Snake:
    """
        A class used to represent each segment of the snake

        Attributes
        ----------
        positioner : int
            represents movement delay between each segments
        color : tuple
            color of the snake in RGB format
        size : int
            snake's size in pixels

        Methods
        -------
        move(inputs)
            encodes inputs array to direction in which snake's head should move
            and append head's new coordinates to global positions array
        update(screen)
            updates segment position on the screen
        body_move(screen)
            gather coordinates from global positions array for the next move for each segment
    """

    positioner = cfg.snake_size // cfg.velocity
    color = (0, 0, 0)
    size = cfg.snake_size

    def __init__(self, number):
        self.number = number
        global positions
        try:
            if self.number == 0:
                raise IndexError
            self.x = positions[-self.number * self.positioner - 1][0]
            self.y = positions[-self.number * self.positioner - 1][1]
        except IndexError:
            self.x = cfg.SCREENWIDTH // 2
            self.y = cfg.SCREENHEIGHT // 2
            positions.clear()
            positions.append((self.x, self.y))
        self.xvelocity = 0
        self.yvelocity = 0
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        self.turn_counter = 0

    def move(self, inputs):
        xtemp = self.xvelocity
        ytemp = self.yvelocity

        if inputs[0] == 1 and self.yvelocity != cfg.velocity:  # pressed[pygame.K_UP]:
            self.yvelocity = -cfg.velocity
            self.xvelocity = 0
        elif inputs[2] == 1 and self.yvelocity != -cfg.velocity:  # pressed[pygame.K_DOWN]:
            self.yvelocity = cfg.velocity
            self.xvelocity = 0
        elif inputs[1] == 1 and self.xvelocity != -cfg.velocity:  # pressed[pygame.K_RIGHT]:
            self.xvelocity = cfg.velocity
            self.yvelocity = 0
        elif inputs[3] == 1 and self.xvelocity != cfg.velocity:  # pressed[pygame.K_LEFT]:
            self.xvelocity = -cfg.velocity
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

        global positions
        positions.append((self.x, self.y))

    def update(self, screen):
        self.rect = pygame.Rect(self.x, self.y, self.size, self.size)
        pygame.draw.rect(screen, self.color, self.rect)

    def body_move(self, screen):
        try:
            self.x = positions[-self.number*self.positioner - 1][0]
            self.y = positions[-self.number*self.positioner - 1][1]
        except IndexError:
            pass
        self.update(screen)
