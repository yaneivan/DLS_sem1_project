import torch.nn as nn
from torchvision.models import resnet34

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.skip(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.relu(x + residual)
    

class HourglassBlock(nn.Module):
  def __init__(self, in_feautres=5, num_features=8):
    super().__init__()
    self.res1 = ResidualBlock(in_feautres, num_features)
    self.res2 = ResidualBlock(num_features, num_features)
    self.res3 = ResidualBlock(num_features, num_features)
    self.res4 = ResidualBlock(num_features, num_features)
    self.resMid = ResidualBlock(num_features, num_features)
    self.conv1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.MaxPool2d(2, 2)
    self.resMid1 = ResidualBlock(num_features, num_features)
    self.resMid2 = ResidualBlock(num_features, num_features)
    self.upconv1 = nn.Upsample(scale_factor=2, mode="nearest")
    self.upconv2 = nn.Upsample(scale_factor=2, mode="nearest")

  def forward(self, x):
    x1 = self.res1(x)
    x = self.conv1(x1)
    x2 = self.res2(x)
    x = self.conv2(x2)
    x = self.resMid(x)
    x1 = self.resMid1(x1)
    x2 = self.resMid2(x2)
    x = self.upconv1(x)
    x += x2
    x = self.res3(x)
    x = self.upconv2(x)
    x += x1 
    x = self.res4(x)
    return x
  
class StackedHourglass(nn.Module):
    def __init__(self, num_outputs, feature_size = 8):
        super().__init__()
        self.first_conv = nn.Conv2d(3, feature_size, kernel_size=1, stride=1, padding=0)
        self.c1 = nn.Conv2d(feature_size, num_outputs, kernel_size=1, stride=1, padding=0)
        self.h1 = HourglassBlock( in_feautres=feature_size, num_features=feature_size)
        self.c2 = nn.Conv2d(feature_size, num_outputs, kernel_size=1, stride=1, padding=0)
        self.h2 = HourglassBlock(in_feautres=feature_size, num_features=feature_size)
        self.c3 = nn.Conv2d(feature_size, num_outputs, kernel_size=1, stride=1, padding=0)
        self.h3 = HourglassBlock(in_feautres=feature_size, num_features=feature_size)
        self.c4 = nn.Conv2d(feature_size, num_outputs, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        outputs = []
        x1 = self.first_conv(x)
        x = self.c1(x1)
        outputs.append(x)

        x1 = self.h1(x1)
        x = self.c2(x1)
        outputs.append(x)

        x1 = self.h2(x1)
        x = self.c3(x1)
        outputs.append(x)

        x1 = self.h3(x1)
        x = self.c4(x1)
        outputs.append(x)
        
        return outputs
    

    
def get_recognition_model(embedding_size=512):
    model = resnet34(weights='DEFAULT') 
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features=512, out_features=512)
    )
    return model