import torch
import torch.nn as nn


class Brain(nn.Module):
    in_nodes = 8
    hidden_nodes = 6
    out_nodes = 4

    def __init__(self):
        super(Brain, self).__init__()
        self.net = nn.Sequential(nn.Linear(self.in_nodes, self.out_nodes),
                                 nn.Sigmoid())

    def forward(self, inputs):
        inputs = torch.tensor(inputs).float()

        output = [0] * 4

        net_product = self.net(inputs).tolist()
        output[net_product.index(max(net_product))] = 1

        return output
