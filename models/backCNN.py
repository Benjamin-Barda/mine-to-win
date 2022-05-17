from torch.nn import Sequential, Conv2d, MaxPool2d, Module, Dropout2d, ELU, AvgPool1d


class BackboneCNN(Module):

    def __init__(self):
        super(BackboneCNN, self).__init__()
        


        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

