import config as cfg
from random import randint
import torch
from brain import Brain


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
        child_first = crossing_over(first_parent, second_parent)
        torch.save(child_first.state_dict(), 'data/' + str(file_number) + '.pt')
        file_number += 1

        child_second = crossing_over(second_parent, first_parent)
        torch.save(child_second.state_dict(), 'data/' + str(file_number) + '.pt')
        file_number += 1

    return file_number


def mating():
    counter = cfg.parents_size
    for it in range(0, cfg.parents_size, 2):
        first = Brain()
        first.load_state_dict(torch.load('data/' + str(it) + '.pt'))
        second = Brain()
        second.load_state_dict(torch.load('data/' + str(it + 1) + '.pt'))
        counter = breeding(first, second, counter)
