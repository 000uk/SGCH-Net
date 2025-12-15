import random
import numpy as np
from torch.utils.data import Dataset
import torch

def augment_skeleton(x):
    # Gaussian Noise only (RGB와 독립)
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.01, x.shape)
        x = x + noise
    return x

class SkeletonDataset(Dataset):
    def __init__(self, data_paths, labels, train=True):
        self.data = data_paths
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.data)
    
    def get_raw(self, idx):
        x = np.load(self.paths[idx])  # (T,21,3)
        label = self.labels[idx]
        return x, label


    def get_clip(self, idx, indices):
        x = np.load(self.data[idx])  # (T, 21, 3)

        # Temporal sampling (공통 indices)
        x = x[indices]

        # 손목 정규화
        wrist = x[:, 0:1, :]
        x = x - wrist

        # Frame Dropout (Skeleton 전용)
        if self.train and random.random() < 0.5:
            drop_idx = random.sample(range(len(indices)), k=random.randint(1, 5))
            x[drop_idx] = 0

        if self.train:
            x = augment_skeleton(x)

        return torch.FloatTensor(x)