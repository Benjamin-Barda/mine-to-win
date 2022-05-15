from colorama import Back
import torch
from torch.autograd import Variable
from torch.nn import Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout2d, GELU
from torch.optim import Adam, SGD

class BackboneCNN(Module):

    def __init__(self):
        super(BackboneCNN, self).__init__()
        
        self.conv1 = Sequential(
            Conv2d(3, 10, kernel_size=5, stride=2),
            GELU(inplace=True),
            Dropout2d(p = 0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv2 = Sequential(
            Conv2d(10, 20, kernel_size=5, stride=2),
            GELU(inplace=True),
            Dropout2d(p = 0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv3 = Sequential(
            
            Conv2d(20, 40, kernel_size=5, stride=2),
            GELU(inplace=True),
            Dropout2d(p = 0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv4 = Sequential(
            
            Conv2d(40, 80, kernel_size=5, stride=2),
            GELU(inplace=True),
            Dropout2d(p = 0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        
    def forward(self, x):
        


class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(4 * 7 * 7, 10)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x