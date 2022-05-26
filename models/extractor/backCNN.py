from torch import nn

# The Ziggurat

class BackboneCNN(nn.Module):
    def __init__(self, is_in_rpn = False):
        super(BackboneCNN, self).__init__()

        # If true stop at the last con layer 
        self.is_in_rpn = is_in_rpn
        
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 20, kernel_size=7, stride=2, padding_mode="replicate", bias=False),
            nn.Mish(True),
            nn.BatchNorm2d(20),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=7, stride=2, padding_mode="replicate", bias = False),
            nn.Mish(True),
            nn.BatchNorm2d(40),
            nn.Dropout2d(p = 0.2, inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(40, 80, kernel_size=3, stride=2, padding_mode="replicate"),
            nn.Mish(True),
            nn.BatchNorm2d(80),
            nn.Dropout2d(p = 0.2, inplace=True),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(80, 120, kernel_size=3, stride=2,  padding_mode="replicate"),
            nn.Mish(True),
            nn.BatchNorm2d(120),
            nn.Dropout2d(p = 0.1, inplace=True),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(120, 3, kernel_size=1, stride=1,  padding_mode="replicate"),
            nn.BatchNorm2d(3),
            nn.Dropout2d(p = 0.1, inplace=True),
        )

        self.pool = nn.AdaptiveMaxPool2d(1)


        
    def forward(self, x, ret=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        if self.is_in_rpn :
            # return after the 4th layer ... no need to go further down the net 
            return x


        x = self.conv5(x)

        k = self.pool(x).reshape((x.shape[0], -1))

        if ret:
            return k, x.clone() # Avg pooling
        
        return k