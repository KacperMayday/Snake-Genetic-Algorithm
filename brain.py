import numpy
import torch
import torch.nn as nn


class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        self.net = nn.Sequential(nn.Linear(8, 6), nn.Sigmoid(), nn.Linear(6, 4), nn.Sigmoid())

    def forward(self, inputs):
        inputs = torch.tensor(inputs).float()

        output = [0] * 4

        net_product = self.net(inputs).tolist()
        output[net_product.index(max(net_product))] = 1

        return output
