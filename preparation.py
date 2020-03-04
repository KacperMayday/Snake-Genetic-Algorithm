"""
    Module responsible to initialize random population

    Run this module to initialize random neural networks for your chosen population size. It is important to create
    networks before running main.py.

    The data will be stored in data/ directory which will be created during the proccess.

    If data directory is already created, whole directory will be erased and replaced with new random samples.
    To prevent your current samples from being replaced, rename your current data/ directory.
"""

from brain import Brain
import torch
from config import population_size
from shutil import rmtree
from os import mkdir


def preparation():
    confirm = input("Do you want to create {} random models? [y/n]".format(population_size))

    if confirm == 'y':
        try:
            rmtree('data')
        except FileNotFoundError:
            pass
        mkdir('data')
        for iterator in range(population_size):
            temp = Brain()
            torch.save(temp.state_dict(), 'data/' + str(iterator) + '.pt')

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
