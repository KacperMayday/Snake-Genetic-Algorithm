"""Main module.

Run this module after preparation.py. If cfg.EPOCH is set to 1, every snake in the
population will be displayed. Otherwise, program is set to training mode in
genetic algorithm.
"""

import statistics as st

import pygame

import config as cfg
import game
import genetics


def constants_validation():
    """Checks if all constant values are valid."""

    if cfg.POPULATION_SIZE % cfg.PARENTS_SIZE != 0:
        raise ValueError('Population size must be a multiplicity of parents size!')

    if cfg.SIZE % cfg.VELOCITY != 0:
        raise ValueError('Size must be a multiplicity of velocity!')

    if cfg.VELOCITY > cfg.SIZE:
        raise ValueError('Velocity must be lower than size!')


def sort_best(score_array):
    """Determines which networks will be used in genetic algorithm.

    This function takes scores list of current generation and returns best
    network's IDs in ascending order.
    
    Parameters
    ----------
    score_array : list
        unsorted list of scores achieved by current generation

    Returns
    -------
    list
        list of best neural networks' IDs
    """
    best_scores = [0] * cfg.PARENTS_SIZE
    for it in range(cfg.PARENTS_SIZE):
        best_scores[it] = scores.index(max(score_array))
        score_array[best_scores[it]] = -1
    best_scores.sort()
    return best_scores


def check_if_win(score_array):
    """Checks win conditions.

    Parameters
    ----------
    score_array : list
        unsorted list of scores achieved by current generation
    """
    if max(score_array) >= cfg.WIN_MAX or st.mean(score_array) >= cfg.WIN_MEAN:
        exit()


def append_stats(score_array):
    """Prints last generation's results and appends it to track.txt.

    Parameters
    ----------
    score_array : list
        unsorted list of scores achieved by current generation
    """
    global best, best_epoch, best_average, best_average, best_average_epoch

    if max(score_array) > best:
        best = max(score_array)
        best_epoch = epoch + 1

    if st.mean(score_array) > best_average:
        best_average = st.mean(score_array)
        best_average_epoch = epoch + 1

    print('Epoch: {}'.format(epoch + 1),
          'Best: {}'.format(max(score_array)),
          'Average: {}'.format(st.mean(score_array)),
          'The Best: {} in epoch {}'.format(best, best_epoch),
          'Best Average: {} in epoch {}'.format(best_average, best_average_epoch))

    with open('data/track.txt', 'a') as file:
        file.write('{};{};{};{};{};{};{}\n'.format(epoch + 1,
                                                   max(score_array),
                                                   st.mean(score_array),
                                                   best,
                                                   best_epoch,
                                                   best_average,
                                                   best_average_epoch))


if __name__ == '__main__':
    constants_validation()  # checks constants
    pygame.init()  # initialize pygame

    best_score = 0
    best_epoch = 0
    best_average = 0
    best_average_epoch = 0

    if cfg.EPOCHS == 1:  # show mode
        loop = input('How many times you want to display last population?')
        try:
            loop = int(loop)
        except ValueError:
            print('Invalid number!')

        for iterator in range(loop):
            scores = []
            for phase in range(cfg.POPULATION_SIZE):
                pygame.display.set_caption('{}/{}'.format(phase + 1, cfg.POPULATION_SIZE))
                screen = pygame.display.set_mode((cfg.SCREENWIDTH, cfg.SCREENHEIGHT))
                current_game = game.Game(phase, screen)
                scores.append(current_game.loop())
            print('Best: {} Average: {}'.format(max(scores), st.mean(scores)))

    else:  # training mode
        for epoch in range(cfg.EPOCHS):
            scores = []
            for phase in range(cfg.POPULATION_SIZE):
                pygame.display.set_caption('{}/{}'.format(phase + 1, cfg.POPULATION_SIZE))
                screen = pygame.display.set_mode((cfg.SCREENWIDTH, cfg.SCREENHEIGHT))
                current_game = game.Game(phase, screen)
                scores.append(current_game.loop())

            append_stats(scores)
            check_if_win(scores)
            best = sort_best(scores)
            genetics.save_best(best)
            genetics.mating()
