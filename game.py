"""Main class responsible for running the game."""

import pygame
import torch

import config as cfg
import genetics
import objects as obj


class Game:
    """Main game class called from main.py.

    This class maintains most of the functions in the game i.e. collisions,
    game loop, scoring and updating the screen. It calls all constructors from
    objects.py and load neural network parameters based on current phase.

    Attributes
    ----------
    phase : int
        represents current phase, used to indicate from which file should
        brain load its parameters.
    screen : obj
        screen object on which game is displayed
    run : bool
        bool value which controls while loop in self.loop()
    score : int
        stores score value but also indicates number of snake's segments
    head : obj
        first of snake's segments, controls the snake movement
    apple : obj
        represents apple, vanishes upon collision with self.head
    body_list : list
        contains all snake's body segments
    sensors : list
        contains all four sensors monitoring each direction
    brain : obj
        neural network object which controls snake's behaviour
    timer : int
        used to measure time passed between two apple catches, prevents snake looping

    Methods
    -------
    update(inputs)
        updates each object currently in the game and moves snake's head
    check()
        checks all game conditions
    get_inputs()
        gather inputs for neural network and returns input list
    loop()
        main game loop, at the end returns final score of the game
    """

    def __init__(self, phase, screen):
        self.phase = phase
        self.screen = screen
        self.run = True
        self.clock = pygame.time.Clock()
        self.score = 0
        self.head = obj.Snake(self.score)
        self.apple = obj.Apple()
        self.body_list = []
        self.sensors = [obj.Sensor('up', self.head.x, self.head.y),
                        obj.Sensor('right', self.head.x, self.head.y),
                        obj.Sensor('down', self.head.x, self.head.y),
                        obj.Sensor('left', self.head.x, self.head.y)]
        self.brain = genetics.Brain()
        self.brain.load_state_dict(torch.load('data/{}.pt'.format(phase)))
        self.brain.eval()
        self.timer = pygame.time.get_ticks()

    def update(self, inputs):
        """Updates game's objects on the screen.

        Sequentially updates each object currently present in the game.
        Body segments are updated from last to first.

        Parameters
        ----------
        inputs : list
            list representing movement orders for snake's head
        """

        for index, body in reversed(list(enumerate(self.body_list[1:], 1))):  
            body.set_coordinates(self.body_list[index-1].x, self.body_list[index-1].y)
            body.update(self.screen)

        try:
            self.body_list[0].set_coordinates(self.head.x, self.head.y)
            self.body_list[0].update(self.screen)
        except IndexError:
            pass

        self.apple.update(self.screen)

        self.head.move(inputs, self.screen)  # moves and updates snake's head

        for sensor in self.sensors:
            sensor.update(self.screen, self.head.x, self.head.y)

    def check(self):
        """Checks all game conditions.

        If collision between snake's head and apple is detected, increments self.score and
        initialize new apple object. If collision between snake's head and its segments or
        screen boundaries is detected, terminates self.loop method. The same thing happens
        when snake fails to catch an apple in time equals IDLE_TIME constant. If so, turn
        counter is set to 0 to eliminate snake's with looping behaviour.
        """

        for body in self.body_list:  # checks body - snake's head collision
            if pygame.Rect.colliderect(self.head.rect, body.rect):  # terminates self.loop() if true
                self.run = False
                break

        if pygame.Rect.colliderect(self.head.rect, self.apple.rect):  # checks apple - snake's head collision
            self.apple = obj.Apple()  # creates new apple object
            self.score += 1  # increment the score
            try:
                self.body_list.append(obj.Snake(self.body_list[-1].x,
                                                self.body_list[-1].y))  # creates new snake's body segment
            except IndexError:
                self.body_list.append(obj.Snake(self.head.x,
                                                self.head.y))

            self.timer = pygame.time.get_ticks()  # resets the timer

        if (self.head.x < 0 or                                     # checks if snake's head is
                self.head.x + self.head.size > cfg.SCREENWIDTH or  # not beyond the screen
                self.head.y < 0 or
                self.head.y + self.head.size > cfg.SCREENHEIGHT):
            self.run = False

        check_time = pygame.time.get_ticks()
        if check_time - cfg.IDLE_TIME > self.timer:  # checks if idle_time is not exceeded
            self.run = False                         # otherwise terminates self.loop()
            self.head.turn_counter = 0               # and reset turn counter

    def get_inputs(self):
        """Gathers input list ready to be passed to the net.

        First four elements in the input list represents the direction to the apple.
        Then for each sensor appends binary value whether it detects collision
        with snake's body segments or if the sensor lies beyond the screen.
        Total inputs: 4 apple directions + 4 sensors * (1 detects snake's body or detects
        screen boundaries) So the length of returned list is 8.

        Note
        ----
        Remember that in pygame x and y are always coordinates for top left pixel!

        Returns
        -------
        inputs : list
            a list of binary inputs representing each detected feature
        """

        inputs = [0] * 4
        if self.apple.y < self.head.y:  # - self.apple.size:    # if apple is higher than head
            inputs[0] = 1
        elif self.apple.y > self.head.y:  # + self.head.size:   # if apple is lower than head
            inputs[1] = 1
        if self.apple.x > self.head.x:  # + self.head.size:     # if apple is on the right
            inputs[2] = 1
        elif self.apple.x < self.head.x:  # - self.apple.size:  # if apple is on the left
            inputs[3] = 1

        for sensor in self.sensors:  # checks collisions for every sensor
            activation = 0
            for body in self.body_list:
                if pygame.Rect.colliderect(sensor.rect, body.rect):
                    activation = 1
                    break
            if activation == 0:
                if (sensor.x <= 0 or sensor.x >= cfg.SCREENWIDTH or   # checks if sensor is beyond the screen
                        sensor.y <= 0 or sensor.y >= cfg.SCREENHEIGHT):
                    activation = 1

            inputs.append(activation)
        return inputs

    def loop(self):
        """Main loop responsible for controlling the game.

        This loop controls game's behaviour and returns final score. It has pause
        functionality on space bar. Mainly, it is responsible to tie game objects
        together.

        Returns
        -------
        self.score : float
            weighted sum of the score and turn counter
        """

        color = (255, 255, 255)  # setting background color
        pause = False

        while self.run:
            self.screen.fill(color)  # fills the screen's background

            for event in pygame.event.get():   # in case of clicking red cross in the corner,
                if event.type == pygame.QUIT:  # exits the game
                    exit()

            pressed = pygame.key.get_pressed()  # detects whether escape or space has been pressed,
            if pressed[pygame.K_ESCAPE]:        # to exit or pause the game
                self.run = False
            if pressed[pygame.K_SPACE]:
                pause = True

            while pause:  # pause loop
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:  # emergency quit
                        exit()

                    if event.type == pygame.KEYDOWN:
                        self.timer = pygame.time.get_ticks()  # resets the timer after pausing
                        pause = False

            inputs = self.brain(self.get_inputs())  # gathers direction order for the snake's head
            self.update(inputs)  # updates objects on the screen
            self.check()  # checks game conditions

            pygame.display.flip()  # flips the frames
            self.clock.tick(cfg.TICKRATE)  # monitor given maximum frame rate

        self.score = self.score + self.head.turn_counter * 0.001  # evaluate the final score
        return self.score
