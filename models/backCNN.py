from torch import nn
import torch

# The Ziggurat v2

class BackboneCNN(nn.Module):
    def __init__(self):
        super(BackboneCNN, self).__init__()
        
        # 381 * 381
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 40, kernel_size=7, stride=2, padding=1, padding_mode="replicate", bias=False),
            nn.Mish(True),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(kernel_size=2, stride=1), # 189 * 189
        )
        # 189 * 189
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 80, kernel_size=7, stride=2, padding=1, padding_mode="replicate", bias = False),
            nn.Mish(True),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.BatchNorm2d(80),
        )
        # 93 * 93
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(80, 120, kernel_size=3, stride=2, padding=1, padding_mode="replicate", bias = False),
            nn.Mish(True),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.BatchNorm2d(120),
        )
        # 47 * 47
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(120, 120, kernel_size=3, bias = False),
            nn.Mish(True),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.BatchNorm2d(120),
        )
        # 47 * 47
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(120, 240, kernel_size=3, stride=2, padding=1, padding_mode="replicate", bias = False),
            nn.Mish(True),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.BatchNorm2d(240),
        )
        # 24 * 24

        self.conv6 = nn.Sequential(
            nn.Conv2d(240, 5, kernel_size=1, bias = False),
            nn.Dropout2d(p = 0.2, inplace=True),
            nn.BatchNorm2d(5),
        )

        self.pool = nn.AdaptiveMaxPool2d(1)
        
        nn.init.kaiming_uniform_(self.conv1[1].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv3[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv4[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv5[0].weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv6[0].weight, nonlinearity='relu')

    def forward(self, x, ret=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        k = self.pool(x).reshape((x.shape[0], -1))

        if ret:
            return k, x.clone() # Avg pooling
        
        return k


# network = BackboneCNN()

# img = torch.rand(1,3,318,318)

# k, a = network.forward(img, True)

# print(a.shape)
