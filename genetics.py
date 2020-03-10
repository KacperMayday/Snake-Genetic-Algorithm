"""Module responsible for handling genetic algorithm functions and neural network"""

from random import randint

import torch
import torch.nn as nn

import config as cfg


def save_best(list_of_bests):
    """Saves best models in the first N files in data directory, where N is
    equal to cfg.PARENTS_SIZE.

    Parameter
    ---------
    list_of_bests : list
        list of best networks' IDs
    """

    for iterator in range(len(list_of_bests)):
        model_file = 'data/{}.pt'.format(list_of_bests[iterator])
        temp = Brain()
        temp.load_state_dict(torch.load(model_file))
        torch.save(temp.state_dict(), 'data/{}.pt'.format(iterator))


def crossing_over(first_parent, second_parent):
    """
    Parameters
    ----------
    first_parent, second_parent : obj
        two neural network models

    Returns
    -------
    child : obj
        new neural network which is the result of combining and mutating parents' networks together
    """

    child = Brain()
    for layer_name, _ in child.named_parameters():
        child_params = child.state_dict()[layer_name]
        first_params = first_parent.state_dict()[layer_name]
        second_params = second_parent.state_dict()[layer_name]
        for tensor in range(len(child_params)):
            try:
                for value in range(len(child_params[tensor])):
                    probability = randint(1, 100)
                    if probability <= cfg.CROSSING_PROBABILITY:
                        child_params[tensor][value] = second_params[tensor][value]
                    else:
                        child_params[tensor][value] = first_params[tensor][value]

            except TypeError:
                probability = randint(1, 100)
                if probability <= cfg.CROSSING_PROBABILITY:
                    child_params[tensor] = second_params[tensor]
                else:
                    child_params[tensor] = first_params[tensor]

        child.state_dict()[layer_name] = child_params
    mutation(child)
    return child


def mutation(model):
    """Adds noise to model's parameters.

    Parameter
    ---------
    model : obj
        neural network which parameters are altered
    """

    for layer_name, _ in model.named_parameters():
        layer_params = model.state_dict()[layer_name]
        for tensor in range(len(layer_params)):
            try:
                for value in range(len(layer_params[tensor])):
                    probability = randint(1, 100)
                    change = randint(-cfg.MUTATION_RATE, cfg.MUTATION_RATE)
                    if probability <= cfg.MUTATION_FREQUENCY:
                        layer_params[tensor][value] = layer_params[tensor][value] \
                                                      + layer_params[tensor][value] \
                                                      * (change / 1000)
            except TypeError:
                probability = randint(1, 100)
                change = randint(-cfg.MUTATION_RATE, cfg.MUTATION_RATE)
                if probability <= cfg.MUTATION_FREQUENCY:
                    layer_params[tensor] = layer_params[tensor] + layer_params[tensor] * (change / 1000)
        model.state_dict()[layer_name] = layer_params


def breeding(first_parent, second_parent, file_number):
    """

    Parameters
    ----------
    first_parent : obj

    second_parent : obj

    file_number : int
    

    Returns
    -------
    file_number : int

    """

    half_offset = (cfg.POPULATION_SIZE - cfg.PARENTS_SIZE) // cfg.PARENTS_SIZE

    for iterator in range(half_offset):
        child_first = crossing_over(first_parent, second_parent)
        torch.save(child_first.state_dict(), 'data/{}.pt'.format(file_number))
        file_number += 1

        child_second = crossing_over(second_parent, first_parent)
        torch.save(child_second.state_dict(), 'data/{}.pt'.format(file_number))
        file_number += 1

    return file_number


def mating():
    """

    """

    counter = cfg.PARENTS_SIZE
    for it in range(0, cfg.PARENTS_SIZE, 2):
        first = Brain()
        first.load_state_dict(torch.load('data/{}.pt'.format(it)))
        second = Brain()
        second.load_state_dict(torch.load('data/{}.pt'.format(it + 1)))
        counter = breeding(first, second, counter)


class Brain(nn.Module):
    """Neural Network class to control snake movement

    Attributes
    ----------
    self.in_nodes : int
        number of input nodes / features for the network
    self.hidden_nodes : int
        number of hidden nodes
    self.out_nodes : int
        number of output nodes, corresponds to four main directions
    self.net : obj


    Methods
    -------
    forward(inputs)
        returns the output array of the network, where each element corresponds to each
        direction: [up, right, down, left]. Maximum value is converted to one and the rest to zero.
    """

    in_nodes = 12
    hidden_nodes = 8
    out_nodes = 4

    def __init__(self):
        super(Brain, self).__init__()
        self.net = nn.Sequential(nn.Linear(self.in_nodes, self.hidden_nodes),
                                 nn.Tanh(),
                                 nn.Linear(self.hidden_nodes, self.out_nodes),
                                 nn.Tanh())

    def forward(self, inputs):
        """

        Parameter
        ---------
        inputs : list

        Returns
        -------
        output : list

        """

        inputs = torch.tensor(inputs).float()

        output = [0] * 4

        net_product = self.net(inputs).tolist()
        output[net_product.index(max(net_product))] = 1

        return output
