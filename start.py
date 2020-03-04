from game import Game
import torch
import config as cfg
import pygame
from brain import Brain
from random import randint
import statistics as st


def save_best(list_of_bests):
    for iterator in range(len(list_of_bests)):
        model_file = 'data/' + str(list_of_bests[iterator]) + '.pt'
        temp = Brain()
        temp.load_state_dict(torch.load(model_file))
        torch.save(temp.state_dict(), 'data/' + str(iterator) + '.pt')


def crossing_over(first_parent, second_parent):
    child = Brain()
    for layer_name, _ in child.named_parameters():
        child_params = child.state_dict()[layer_name]
        first_params = first_parent.state_dict()[layer_name]
        second_params = second_parent.state_dict()[layer_name]
        for tensor in range(len(child_params)):
            try:
                for value in range(len(child_params[tensor])):
                    probability = randint(1, 100)
                    if probability <= cfg.crossing_probability:
                        child_params[tensor][value] = second_params[tensor][value]
                    else:
                        child_params[tensor][value] = first_params[tensor][value]
            except TypeError:
                probability = randint(1, 100)
                if probability <= cfg.crossing_probability:
                    child_params[tensor] = second_params[tensor]
                else:
                    child_params[tensor] = first_params[tensor]

        child.state_dict()[layer_name] = child_params
    mutation(child)
    return child


def mutation(model):
    for layer_name, _ in model.named_parameters():
        layer_params = model.state_dict()[layer_name]
        for tensor in range(len(layer_params)):
            try:
                for value in range(len(layer_params[tensor])):
                    probability = randint(1, 100)
                    change = randint(-cfg.mutation_rate, cfg.mutation_rate)
                    if probability <= cfg.mutation_frequency:
                        layer_params[tensor][value] = layer_params[tensor][value] \
                                                      + layer_params[tensor][value] \
                                                      * (change / 1000)
            except TypeError:
                probability = randint(1, 100)
                change = randint(-cfg.mutation_rate, cfg.mutation_rate)
                if probability <= cfg.mutation_frequency:
                    layer_params[tensor] = layer_params[tensor] + layer_params[tensor] * (change / 1000)
        model.state_dict()[layer_name] = layer_params


def breeding(first_parent, second_parent, file_number):
    half_offset = (cfg.population_size - cfg.parents_size) // cfg.parents_size
    for iterator in range(half_offset):
        child = crossing_over(first_parent, second_parent)
        torch.save(child.state_dict(), 'data/' + str(file_number) + '.pt')
        file_number += 1
    for iterator in range(half_offset):
        child = crossing_over(second_parent, first_parent)
        torch.save(child.state_dict(), 'data/' + str(file_number) + '.pt')
        file_number += 1
    return file_number


def check_errors():
    if cfg.population_size % cfg.parents_size != 0:
        raise ValueError('Population size must be a multiplicity of parents size!')


if __name__ == "__main__":
    check_errors()
    pygame.init()

    thebest = 0
    best_epoch = 0
    best_average = 0
    best_average_epoch = 0

    for epoch in range(cfg.epoch_number):
        scores = []
        for phase in range(cfg.population_size):
            pygame.display.set_caption('{}/{}'.format(str(phase + 1), str(cfg.population_size)))
            game = Game(phase)
            scores.append(game.score)

        if max(scores) > thebest:
            thebest = max(scores)
            best_epoch = epoch + 1

        if st.mean(scores) > best_average:
            best_average = st.mean(scores)
            best_average_epoch = epoch + 1

        print('Epoch: {}'
              ' Best: {}'
              ' Average: {}'
              ' The Best: {} in epoch {}'
              ' Best Average: {} in epoch {}'.format(epoch + 1,
                                                     max(scores),
                                                     st.mean(
                                                         scores),
                                                     thebest,
                                                     best_epoch,
                                                     best_average,
                                                     best_average_epoch))
        with open('data/track.txt', 'a') as file:
            file.write('{};{};{};{};{};{};{}\n'.format(epoch + 1,
                                                       max(scores),
                                                       st.mean(scores),
                                                       thebest,
                                                       best_epoch,
                                                       best_average,
                                                       best_average_epoch))
        # exit()
        if max(scores) >= 100 or st.mean(scores) >= 50:
            exit()

        best = [0] * cfg.parents_size
        for it in range(cfg.parents_size):
            best[it] = scores.index(max(scores))
            scores[best[it]] = -1
        best.sort()

        save_best(best)

        counter = cfg.parents_size
        for it in range(0, cfg.parents_size, 2):
            first = Brain()
            first.load_state_dict(torch.load('data/' + str(it) + '.pt'))
            second = Brain()
            second.load_state_dict(torch.load('data/' + str(it + 1) + '.pt'))
            counter = breeding(first, second, counter)
