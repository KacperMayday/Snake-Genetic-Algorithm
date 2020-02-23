# Module responsible to initialize random weights' files
from brain import Brain
import torch
from config import population_size


# confirm = input("Do you want to create {} random models? [y/n]".format(population_size))
confirm = 'y'
if confirm == 'y':
    for iterator in range(population_size):
        temp = Brain()
        torch.save(temp.state_dict(), 'data/'+str(iterator))
    print('Process done')
