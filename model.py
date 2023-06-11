from torch import nn

# import dropblock


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=(2, 1), padding=(4, 2))  # play with stride
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.dropOut1 = nn.Dropout2d(.1)
        # self.dropOut1 = nn.DropBlock2D(.1)
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)
        self.res1 = ResBlock(64, 64, 1)  # stride
        self.res2 = ResBlock(64, 128, 2)
        self.res3 = ResBlock(128, 256, 2)
        self.res4 = ResBlock(256, 512, 2)
        self.avgPool = nn.AvgPool2d(10)
        self.dropOut2 = nn.Dropout(.2)
        self.fc1 = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()#BCEWithLogicsLoss

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropOut1(x)
        x = self.pool1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.avgPool(x)
        x = x.flatten(start_dim=1)
        x = self.dropOut2(x)
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

        x = self.conv2(x)
        x = self.bn2(x)

        x_input = self.conv1x1(x_input)
        x_input = self.bn1x1(x_input)
        x = x + x_input
        x = self.relu2(x)
        return x
