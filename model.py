from torch import nn


# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self):
        """
        Initializes the ResNet model.

        The model architecture consists of convolutional layers, batch normalization,
        ReLU activation functions, dropout layers, max-pooling, residual blocks, and fully
        connected layers.

        """
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(
            3, 64, 7, stride=(2, 1), padding=(4, 2)
        )  # Initial convolution layer
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization
        self.relu1 = nn.ReLU()  # ReLU activation
        self.dropOut1 = nn.Dropout2d(0.1)  # 2D dropout layer
        self.pool1 = nn.MaxPool2d(3, 2, padding=1)  # Max pooling
        self.res1 = ResBlock(64, 64, 1)  # First residual block
        self.res2 = ResBlock(64, 128, 2)  # Second residual block
        self.res3 = ResBlock(128, 256, 2)  # Third residual block
        self.res4 = ResBlock(256, 512, 2)  # Fourth residual block
        self.avgPool = nn.AvgPool2d(10)  # Average pooling
        self.dropOut2 = nn.Dropout(0.2)  # Dropout layer
        self.fc1 = nn.Linear(512, 2)  # Fully connected layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        """
        Defines the forward pass of the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
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


# Define the Residual Block class
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        Initializes a residual block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the convolutional layers.

        """
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 convolution for adapting input size and channels
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.bn1x1 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of a residual block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_input = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x_input = self.conv1x1(x_input)
        x_input = self.bn1x1(x_input)
        x = x + x_input  # Skip connection
        x = self.relu2(x)
        return x
