from torch import nn

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)
        self.res1 = ResBlock(128, 128, 1)
        self.res2 = ResBlock(128, 128, 2)
        self.res3 = ResBlock(128, 256, 2)
        self.res4 = ResBlock(256, 512, 2)
        self.avgPool = nn.AvgPool2d(10)
        self.fc1 = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avgPool(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.relu4 = nn.ReLU()
        # the input of the ResBlock is added to the output. Therefore, the size and number
        # of channels needs to be adapted. To this end, we recommend to apply a 1  1 convolution to
        # the input with stride and channels set accordingly.
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU()

    def forward(self, x):
        x_input = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x_1 = x

        x = x_1 + x_input
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x_2 = x

        x = x_2 + x_input
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x_3 = x

        x = x_3 + x_input
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x_4 = x

        x_input = self.conv1x1(x_input)
        x_input = self.bn1x1(x_input)
        x = x + x_1 + x_2 + x_3 + x_4
        x = x + x_input
        x = self.relu2(x)
        return x
