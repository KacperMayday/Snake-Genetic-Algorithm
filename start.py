import statistics as st

import pygame

import config as cfg
import game
import genetics


def constants_validation():
    if cfg.POPULATION_SIZE % cfg.PARENTS_SIZE != 0:
        raise ValueError('Population size must be a multiplicity of parents size!')

    if cfg.SIZE % cfg.VELOCITY != 0:
        raise ValueError('Size must be a multiplicity of velocity!')

    if cfg.VELOCITY > cfg.SIZE:
        raise ValueError('Velocity must be lower than size!')


def sort_best(score_array):
    best_scores = [0] * cfg.PARENTS_SIZE
    for it in range(cfg.PARENTS_SIZE):
        best_scores[it] = scores.index(max(score_array))
        score_array[best_scores[it]] = -1
    best_scores.sort()
    return best_scores


def check_if_win(score_array):
    if max(score_array) >= cfg.WIN_MAX or st.mean(score_array) >= cfg.WIN_MEAN:
        exit()


def append_stats(score_array):
    global thebest, best_epoch, best_average, best_average, best_average_epoch

    if max(score_array) > thebest:
        thebest = max(score_array)
        best_epoch = epoch + 1

    if st.mean(score_array) > best_average:
        best_average = st.mean(score_array)
        best_average_epoch = epoch + 1

    print('Epoch: {}'
          ' Best: {}'
          ' Average: {}'
          ' The Best: {} in epoch {}'
          ' Best Average: {} in epoch {}'.format(epoch + 1,
                                                 max(score_array),
                                                 st.mean(score_array),
                                                 thebest,
                                                 best_epoch,
                                                 best_average,
                                                 best_average_epoch))
    with open('data/track.txt', 'a') as file:
        file.write('{};{};{};{};{};{};{}\n'.format(epoch + 1,
                                                   max(score_array),
                                                   st.mean(score_array),
                                                   thebest,
                                                   best_epoch,
                                                   best_average,
                                                   best_average_epoch))


if __name__ == '__main__':
    constants_validation()
    pygame.init()

    thebest = 0
    best_epoch = 0
    best_average = 0
    best_average_epoch = 0

    for epoch in range(cfg.EPOCHS):
        scores = []
        for phase in range(cfg.POPULATION_SIZE):
            pygame.display.set_caption('{}/{}'.format(phase + 1, cfg.POPULATION_SIZE))
            screen = pygame.display.set_mode((cfg.SCREENWIDTH, cfg.SCREENHEIGHT))
            current_game = game.Game(phase, screen)
            scores.append(current_game.score)

        append_stats(scores)
        check_if_win(scores)
        best = sort_best(scores)
        genetics.save_best(best)
        genetics.mating()
