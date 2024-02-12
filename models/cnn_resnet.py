import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet512(nn.Module):
    def __init__(self, res_layers = [2, 2, 2, 2]):
        super().__init__()
        
        def ResidualBlock (in_channels, out_channels, kernel_size):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size, padding='same')
            )
            
        self.conv1 = nn.Conv2d(3, 64, 3, padding='same')
        self.Res1_modules = nn.ModuleList([ResidualBlock(64, 64, 3) for _ in range(res_layers[0])])
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding='same')
        self.Res2_modules = nn.ModuleList([ResidualBlock(128, 128, 3) for _ in range(res_layers[1])])
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding='same')
        self.Res3_modules = nn.ModuleList([ResidualBlock(256, 256, 3) for _ in range(res_layers[2])])
        
        self.conv4 = nn.Conv2d(256, 512, 3, padding='same')
        self.Res4_modules = nn.ModuleList([ResidualBlock(512, 512, 3) for _ in range(res_layers[3])])
        
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        h = x
        for module in self.Res1_modules:
            x = F.relu(module(x) + h)
            h = x
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        h = x
        for module in self.Res2_modules:
            x = F.relu(module(x) + h)
            h = x
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        h = x
        for module in self.Res3_modules:
            x = F.relu(module(x) + h)
            h = x
        x = self.pool(x)
        
        x = F.relu(self.conv4(x))
        h = x
        for module in self.Res4_modules:
            x = F.relu(module(x) + h)
            h = x
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x