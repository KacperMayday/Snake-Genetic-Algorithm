# Module responsible to initialize random weights' files
from brain import Brain
import torch
from config import population_size
from shutil import rmtree
from os import mkdir

# confirm = input("Do you want to create {} random models? [y/n]".format(population_size))
confirm = 'y'
if confirm == 'y':
    try:
        rmtree('data')
    except FileNotFoundError:
        pass
    mkdir('data')
    for iterator in range(population_size):
        temp = Brain()
        torch.save(temp.state_dict(), 'data/'+str(iterator))

    with open('data/track.txt', 'w') as file:
        file.write('')

    with open('config.py', 'r') as config:
        with open('data/track.txt', 'a') as track:
            for line in config:
                track.write(line)
            track.write('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<Training>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
            track.write('\nEpoch;Best;Average;TheBestScore;BestEpoch;BestAverage;BestAverageEpoch\n')
    print('Process done')
