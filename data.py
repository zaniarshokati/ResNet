from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

# Mean and standard deviation values for data normalization
train_mean = [0.59685254 * 0.01, 0.59685254 * 0.01, 0.59685254 * 0.01]
train_std = [0.16043035 * 0.01, 0.16043035 * 0.01, 0.16043035 * 0.01]


# Custom dataset class
class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        """
        Initializes a ChallengeDataset instance.

        Args:
            data (object): The dataset containing file information and labels.
            mode (str): Dataset mode, either "val" or "train".

        Raises:
            AssertionError: If an invalid mode is provided.
        """
        self.data = data
        assert mode == "val" or mode == "train", "invalid mode argument"
        self.mode = mode
        self._transform = tv.transforms.Compose(
            [
                tv.transforms.ToPILImage(),
                tv.transforms.RandomVerticalFlip(),
                tv.transforms.RandomHorizontalFlip(),
                tv.transforms.RandomRotation(5),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(train_mean, train_std),
                tv.transforms.ColorJitter(),
            ]
        )

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image and label tensors.
        """
        path = Path(self.data.filename[index])
        image = imread(path)
        image = self._transform(gray2rgb(image))
        label = torch.Tensor([self.data.crack[index], self.data.inactive[index]])
        sample = image, label
        return sample
