from torch.utils.data import Dataset
import torch
from pathlib import Path
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254*0.01, 0.59685254*0.01, 0.59685254*0.01]
train_std = [0.16043035*0.01, 0.16043035*0.01, 0.16043035*0.01]


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, data, mode):
        self.data = data
        assert mode == 'val' or mode == 'train', 'invalid mode argument'
        self.mode = mode
        self._transform = tv.transforms.Compose(
            [tv.transforms.ToPILImage(),
             tv.transforms.RandomVerticalFlip(), tv.transforms.RandomHorizontalFlip(),
             tv.transforms.RandomRotation(5),  # RandomRotation
             tv.transforms.ToTensor(), tv.transforms.Normalize(train_mean, train_std),tv.transforms.ColorJitter()])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = Path(self.data.filename[index])
        image = imread(path)
        image = self._transform(gray2rgb(image))
        label = torch.Tensor([self.data.crack[index], self.data.inactive[index]])
        sample = image, label
        return sample
