from torch import nn

class BackboneCNN(nn.Module):
    def __init__(self):
        super(BackboneCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=2, padding_mode="replicate"),
            nn.Mish(True),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3, stride=2, padding_mode="replicate"),
            nn.Mish(True),
            nn.Dropout2d(p = 0.2, inplace=True),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 40, kernel_size=3, stride=2, padding_mode="replicate"),
            nn.Mish(True),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=3, stride=1, padding="same", padding_mode="replicate"),
            nn.Mish(True),
            nn.Dropout2d(p = 0.1, inplace=True),
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(40, 80, kernel_size=5, stride=1, padding="same", padding_mode="replicate")
        )

        self.pool = nn.AvgPool2d(51, ceil_mode=True)

        
    def forward(self, x, ret=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if ret:
            return self.pool(x).reshape(x.shape[0], x.shape[1]), x.clone() # Avg pooling
        
        return self.pool(x).reshape(x.shape[0], x.shape[1])
