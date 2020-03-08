import config as cfg
import pygame
import torch
from snake import Snake
from apple import Apple
from sensor import Sensor
from brain import Brain


class Game:
    screen = pygame.display.set_mode((cfg.SCREENWIDTH, cfg.SCREENHEIGHT))
    clock = pygame.time.Clock()

    def __init__(self, phase):
        self.run = True
        self.score = 0
        self.head = Snake(self.score)
        self.apple = Apple()
        self.body_list = []
        self.sensors = [Sensor('up'), Sensor('right'), Sensor('down'), Sensor('left')]
        self.pause = False
        self.end = False
        self.brain = Brain()
        self.phase = phase
        self.brain.load_state_dict(torch.load('data/'+str(phase) + '.pt'))
        self.brain.eval()
        self.timer = pygame.time.get_ticks()
        self.loop()

    def update(self):
        self.head.update(self.screen)
        for body in self.body_list:
            body.body_move(self.screen)
        self.apple.update(self.screen)
        for sensor in self.sensors:
            sensor.update(self.screen)

    def check(self):
        if pygame.Rect.colliderect(self.head.rect, self.apple.rect):
            self.apple = Apple()
            self.score += 1
            self.body_list.append(Snake(self.score))
            self.timer = pygame.time.get_ticks()

        for body in self.body_list[1:]:
            if pygame.Rect.colliderect(self.head.rect, body.rect):
                self.run = False
                break

        if self.head.x <= 0 or \
                self.head.x + cfg.snake_size >= cfg.SCREENWIDTH or \
                self.head.y <= 0 or \
                self.head.y + cfg.snake_size >= cfg.SCREENHEIGHT:
            self.run = False

        check_time = pygame.time.get_ticks()
        if check_time - cfg.idle_time > self.timer:
            self.run = False
            self.head.turn_counter = 0

    def get_inputs(self):
        inputs = [0] * 4
        if self.apple.y <= self.head.y - cfg.apple_size:
            inputs[0] = 1
        elif self.apple.y >= self.head.y + cfg.snake_size:
            inputs[1] = 1
        if self.apple.x >= self.head.x + cfg.snake_size:
            inputs[2] = 1
        elif self.apple.x <= self.head.x - cfg.apple_size:
            inputs[3] = 1

        for sensor in self.sensors:
            activation = 0
            for body in self.body_list[1:]:
                if pygame.Rect.colliderect(sensor.rect, body.rect):
                    activation = 1
                    break

            if sensor.x <= 0 or sensor.x >= cfg.SCREENWIDTH or sensor.y <= 0 or sensor.y >= cfg.SCREENHEIGHT:
                activation = 1

            inputs.append(activation)
        return inputs

    def get_distance(self):
        distance = pow(pow(self.apple.y-self.head.y, 2) + pow(self.apple.x-self.head.x, 2), 0.5)
        return distance

    def loop(self):
        color = (255, 255, 255)
        while self.run:
            self.screen.fill(color)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_ESCAPE]:
                self.end = True
                self.run = False
            if pressed[pygame.K_SPACE]:
                self.pause = True
            while self.pause:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                    if event.type == pygame.KEYDOWN:
                        self.pause = False

            inputs = self.brain(self.get_inputs())
            self.head.move(inputs)
            self.check()
            self.update()

            pygame.display.flip()
            self.clock.tick(cfg.TICKRATE)

        self.score = self.score + self.head.turn_counter * 0.001
