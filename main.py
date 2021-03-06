"""Main module.

Run this module after preparation.py. If cfg.EPOCH is set to 1, every snake in the
population will be displayed. Otherwise, program is set to training mode in
genetic algorithm.
"""

import statistics as st
import argparse

import pygame

import config as cfg
import game
import genetics
import preparation as prep


def constants_validation():
    """Checks if all constant values are valid."""

    if cfg.POPULATION_SIZE % cfg.PARENTS_SIZE != 0:
        raise ValueError('Population size must be a multiplicity of parents size!')


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
    best_scores : list
        list of best neural networks' IDs
    """

    best_scores = [0] * cfg.PARENTS_SIZE
    for it in range(cfg.PARENTS_SIZE):
        best_scores[it] = score_array.index(max(score_array))
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


def append_stats(score_array, epoch, best_score, best_epoch, best_average, best_average_epoch):
    """Prints last generation's results and appends it to track.txt.

    Parameters
    ----------
    score_array : list
        unsorted list of scores achieved by current generation
    epoch : int
        epoch number
    best_score : int
        best score overall
    best_epoch : int
        epoch in which best score has been achieved
    best_average : int
        best average score in generation
    best_average_epoch : int
        epoch in which best average score has been achieved
    """

    best = round(max(score_array), 3)
    average = round(st.mean(score_array), 3)

    print('Epoch: {:0>3d}'.format(epoch + 1),
          'Best: {:7.3f}'.format(best),
          'Average: {:7.3f}'.format(average),
          'The Best: {:7.3f} in epoch {:0>3d}'.format(best_score, best_epoch),
          'Best Average: {:7.3f} in epoch {:0>3d}'.format(best_average, best_average_epoch))

    with open('data/track.txt', 'a') as file:
        file.write('{};{};{};{};{};{};{}\n'.format(epoch + 1,
                                                   best,
                                                   average,
                                                   best_score,
                                                   best_epoch,
                                                   best_average,
                                                   best_average_epoch))


def run_generation():
    """Runs all individuals in the population.

    Returns
    -------
    score_array : list
        list of scores achieved by each neural network in generation.
    """
    score_array = []
    for phase in range(cfg.POPULATION_SIZE):
        pygame.display.set_caption('{}/{}'.format(phase + 1, cfg.POPULATION_SIZE))
        screen = pygame.display.set_mode((cfg.SCREENWIDTH, cfg.SCREENHEIGHT))
        current_game = game.Game(phase, screen)
        score_array.append(current_game.loop())
    return score_array


def show_mode(loop):
    """Show mode without genetic algorithm. Used only for displaying the results"""
    '''loop = input('How many times you want to display last population?\n')
    try:
        loop = int(loop)
    except ValueError:
        print('Invalid number!')'''

    for iterator in range(loop):
        scores = run_generation()

        print('Best: {} Average: {}'.format(max(scores), st.mean(scores)))


def training_mode():
    """Training mode with genetic algorithm applied. Used to train the population."""
    best_score = 0
    best_epoch = 0
    best_average = 0
    best_average_epoch = 0

    for epoch in range(cfg.EPOCHS):
        scores = run_generation()
        if max(scores) > best_score:
            best_score = round(max(scores), 3)
            best_epoch = epoch + 1

        if st.mean(scores) > best_average:
            best_average = round(st.mean(scores), 3)
            best_average_epoch = epoch + 1

        append_stats(scores, epoch, best_score, best_epoch, best_average, best_average_epoch)
        check_if_win(scores)
        best = sort_best(scores)
        genetics.save_best(best)
        genetics.mating()


if __name__ == '__main__':
    constants_validation()  # checks constants

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--show", help="open show mode", type=int)
    parser.add_argument("-r", "--restart", help="prepare population", action="store_true")
    args = parser.parse_args()

    pygame.init()  # initialize pygame

    if args.restart:
        print("Preparation...")
        prep.preparation()
        print("Preparation complete")
    if args.show:
        print("Show mode...")
        show_mode(args.show)
    else:
        print("Training mode...")
        training_mode()
