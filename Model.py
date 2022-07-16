# Imports
import torch.nn as nn


""" Example model to be used for the Dsprite dataset """
class Dsprite_network(nn.Module):
    """
    6 layer neural network with:
        3 layers of conv --> batchnorm --> Relu
        1 maxplooling after convs
        2 layers of linear --> Relu
        1 layer of linear --> softmax
    
    """

    def __init__(self, output_channels):
        """
        Init for the model
        Args:
            output_channels (int): nr of output channels (classes) for the model
        """
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=8, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=6, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=5,stride = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
            nn.Linear(2048,128),
            nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(
            nn.Linear(64, output_channels),
            nn.Softmax(dim = 1))

        self.conv_network = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            nn.MaxPool2d(2),
        )
        self.linear_network = nn.Sequential(
            self.layer4,
            self.layer5,
            self.layer6
        )

    def forward(self, x):
        """
        Forward function for the model
        """
        batch_size = x.shape[0]
        x = self.conv_network(x).reshape(batch_size, -1)
        x = self.linear_network(x)
        return x