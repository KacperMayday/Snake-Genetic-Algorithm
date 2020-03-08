"""Module responsible to initialize random population

Run this module to initialize random neural networks for your chosen population size. It is important to create
networks before running main.py.

The data will be stored in data/ directory which will be created during the proccess.

If data directory is already created, whole directory will be erased and replaced with new random samples.
To prevent your current samples from being replaced, rename your current data/ directory.
"""

from os import mkdir
from shutil import rmtree

import torch

import config as cfg
import genetics


def preparation():
    # confirm = input('Do you want to create %i random models? [y/n]' % cfg.POPULATION_SIZE)
    confirm = 'y'  # for testing
    if confirm == 'y':
        try:
            rmtree('data')
        except FileNotFoundError:
            pass
        mkdir('data')
        for iterator in range(cfg.POPULATION_SIZE):
            temp = genetics.Brain()
            torch.save(temp.state_dict(), 'data/{}.pt'.format(iterator))

        with open('data/track.txt', 'w') as file:
            file.write('')

        with open('config.py', 'r') as config:
            with open('data/track.txt', 'a') as track:
                for line in config:
                    track.write(line)
                track.write('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
                track.write('\nEpoch;Best;Average;TheBestScore;BestEpoch;BestAverage;BestAverageEpoch\n')
        print('Process done')


if __name__ == '__main__':
    preparation()
