import torch
from .spectral_normalization import SpectralNorm
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        # Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv_1 = SpectralNorm(nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1))
        self.conv_2 = SpectralNorm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1))
        self.bn_1 = nn.BatchNorm2d(256)
        self.conv_3 = SpectralNorm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1))
        self.bn_2 = nn.BatchNorm2d(512)
        self.conv_4 = SpectralNorm(nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1))
        self.bn_3 = nn.BatchNorm2d(1024)
        self.conv_5 = SpectralNorm(nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        ##########       END      ##########

    def forward(self, x):
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = self.leaky_relu(self.conv_1(x))
        x = self.leaky_relu(self.bn_1(self.conv_2(x)))
        x = self.leaky_relu(self.bn_2(self.conv_3(x)))
        x = self.leaky_relu(self.bn_3(self.conv_4(x)))
        x = self.conv_5(x)
        ##########       END      ##########
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv_1 = nn.ConvTranspose2d(self.noise_dim, 1024, kernel_size=4, stride=1)
        self.bn_1 = nn.BatchNorm2d(1024)
        self.conv_2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.bn_2 = nn.BatchNorm2d(512)
        self.conv_3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)
        self.conv_4 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn_4 = nn.BatchNorm2d(128)
        self.conv_5 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        ##########       END      ##########

    def forward(self, x):
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = F.relu(self.bn_4(self.conv_4(x)))
        x = torch.tanh(self.conv_5(x))
        ##########       END      ##########

        return x
