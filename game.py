import pygame
import torch

import config as cfg
import genetics
import objects as obj


class Game:
    clock = pygame.time.Clock()

    def __init__(self, phase, screen):
        self.run = True
        self.screen = screen
        self.score = 0
        self.head = obj.Snake(self.score)
        self.apple = obj.Apple()
        self.body_list = []
        self.sensors = [obj.Sensor('up'), obj.Sensor('right'), obj.Sensor('down'), obj.Sensor('left')]
        self.brain = genetics.Brain()
        self.phase = phase
        self.brain.load_state_dict(torch.load('data/{}.pt'.format(phase)))
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
            self.apple = obj.Apple()
            self.score += 1
            self.body_list.append(obj.Snake(self.score))
            self.timer = pygame.time.get_ticks()

        for body in self.body_list[1:]:
            if pygame.Rect.colliderect(self.head.rect, body.rect):
                self.run = False
                break

        if (self.head.x <= 0 or
                self.head.x + self.head.size >= cfg.SCREENWIDTH or
                self.head.y <= 0 or
                self.head.y + self.head.size >= cfg.SCREENHEIGHT):
            self.run = False

        check_time = pygame.time.get_ticks()
        if check_time - cfg.IDLE_TIME > self.timer:
            self.run = False
            self.head.turn_counter = 0

    def get_inputs(self):
        inputs = [0] * 4
        if self.apple.y <= self.head.y - self.apple.size:
            inputs[0] = 1
        elif self.apple.y >= self.head.y + self.head.size:
            inputs[1] = 1
        if self.apple.x >= self.head.x + self.head.size:
            inputs[2] = 1
        elif self.apple.x <= self.head.x - self.apple.size:
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

    def loop(self):
        color = (255, 255, 255)
        pause = False
        while self.run:
            self.screen.fill(color)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()

            pressed = pygame.key.get_pressed()
            if pressed[pygame.K_ESCAPE]:
                self.run = False
            if pressed[pygame.K_SPACE]:
                pause = True
            while pause:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        exit()
                    if event.type == pygame.KEYDOWN:
                        self.timer = pygame.time.get_ticks()
                        pause = False

            inputs = self.brain(self.get_inputs())
            self.head.move(inputs)
            self.check()
            self.update()

            pygame.display.flip()
            self.clock.tick(cfg.TICKRATE)

        self.score = self.score + self.head.turn_counter * 0.001
