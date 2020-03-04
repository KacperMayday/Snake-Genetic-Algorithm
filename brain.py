import torch
import torch.nn as nn


class Brain(nn.Module):
    """
        Neural Network class to control snake movement

        Attributes
        ----------
        in_nodes : int
            number of input nodes / features for the network
        hidden_nodes : int
            number of hidden nodes
        out_nodes : int
            number of output nodes, corresponds to four main directions

        Methods
        -------
        forward(inputs)
            returns the output array of the network, where each element corresponds to each
            direction: [up, right, down, left]. Maximum value is converted to one and the rest to zero.
    """

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
