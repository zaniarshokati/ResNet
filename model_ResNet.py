import tensorflow as tf
from tensorflow.keras import layers

class ResNet(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = layers.Conv2D(64, 7, strides=(2, 1), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.dropOut1 = layers.Dropout(0.1)
        self.pool1 = layers.MaxPool2D(3, strides=2, padding='same')
        self.res1 = ResBlock(64, 64, 1)
        self.res2 = ResBlock(64, 128, 2)
        self.res3 = ResBlock(128, 256, 2)
        self.res4 = ResBlock(256, 512, 2)
        self.avgPool = layers.AveragePooling2D(10)
        self.dropOut2 = layers.Dropout(0.2)
        self.fc1 = layers.Dense(2, activation='linear')
        self.sigmoid = layers.Activation('sigmoid')

    def call(self, x):
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
        x = tf.keras.layers.Flatten()(x)
        x = self.dropOut2(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x


class ResBlock(tf.keras.Model):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, 3, strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(out_channels, 3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv1x1 = layers.Conv2D(out_channels, 1, strides=stride)
        self.bn1x1 = layers.BatchNormalization()

        self.relu2 = layers.ReLU()

    def call(self, x):
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
