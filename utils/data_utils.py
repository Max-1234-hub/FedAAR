import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import numpy as np
from torch.utils.data import Dataset
import os

class DigitsDataset(Dataset):
    def __init__(self, data_path, percent=0.1, train=True, transform=None):
        if train:
            if percent >= 0.1:
                for part in range(int(percent*10)):
                    if part == 0:
                        self.images, self.labels = np.load(os.path.join(data_path, 'train_part{}.pkl'.format(part)), allow_pickle=True)
                    else:
                        images, labels = np.load(os.path.join(data_path, 'train_part{}.pkl'.format(part)), allow_pickle=True)
                        self.images = np.concatenate([self.images,images], axis=0)
                        self.labels = np.concatenate([self.labels,labels], axis=0)
            else:
                self.images, self.labels = np.load(os.path.join(data_path, 'train_part0.pkl'), allow_pickle=True)
                data_len = int(self.images.shape[0] * percent*10)
                self.images = self.images[:data_len]
                self.labels = self.labels[:data_len]
        else:
            self.images, self.labels = np.load(os.path.join(data_path, 'test.pkl'), allow_pickle=True)

        self.transform = transform
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label