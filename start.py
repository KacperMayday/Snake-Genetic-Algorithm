from game import Game
import config as cfg
import pygame
import statistics as st
import genetics


def check_errors():
    if cfg.population_size % cfg.parents_size != 0:
        raise ValueError('Population size must be a multiplicity of parents size!')


def sort_best(score_array):
    best_scores = [0] * cfg.parents_size
    for it in range(cfg.parents_size):
        best_scores[it] = scores.index(max(score_array))
        score_array[best_scores[it]] = -1
    best_scores.sort()
    return best_scores


def check_if_win(score_array):
    if max(score_array) >= 100 or st.mean(score_array) >= 50:
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

        append_stats(scores)
        check_if_win(scores)
        best = sort_best(scores)
        genetics.save_best(best)
        genetics.mating()
