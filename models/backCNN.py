from torch.nn import Sequential, Conv2d, MaxPool2d, Module, Dropout2d, GELU, MaxPool1d


class BackboneCNN(Module):

    def __init__(self):
        super(BackboneCNN, self).__init__()
        
        self.conv1 = Sequential(
            Conv2d(3, 10, kernel_size=5, stride=1),
            GELU(),
            Dropout2d(p = 0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = Sequential(
            Conv2d(10, 20, kernel_size=5, stride=1),
            GELU(),
            Dropout2d(p = 0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = Sequential(
            
            Conv2d(20, 40, kernel_size=5, stride=1),
            GELU(),
            Dropout2d(p = 0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = Sequential(
            Conv2d(40, 80, kernel_size=5, stride=1),
            GELU(),
            Dropout2d(p = 0.2, inplace=True),
            MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv5 = Sequential(
            
            Conv2d(80, 120, kernel_size=5, stride=1),
            GELU(),
            Dropout2d(p = 0.2, inplace=True)
        )
        
        self.pooling = MaxPool1d(kernel_size = 5, stride = 5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = self.pooling(x)
        _, _, c = x.size()

        return x.permute(0, 2, 1).view(n, c, w, h)

