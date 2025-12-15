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

class SkeletonDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, labels, train=True):
        self.data = data_paths
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.data)

    # raw data만 가져오는 헬퍼
    def _get_data(self, idx):
        return np.load(self.data[idx])  # (T, 21, 3)

    # multimodal 용 clip
    def get_clip(self, idx, indices):
        x = self._get_data(idx)  # (T, 21, 3)
        x = x[indices]  # temporal sampling

        # 손목 정규화
        wrist = x[:, 0:1, :]
        x = x - wrist

        # Frame Dropout (train 전용)
        if self.train and random.random() < 0.5:
            drop_idx = random.sample(range(len(indices)), k=random.randint(1, 5))
            x[drop_idx] = 0

        # 데이터 증강
        if self.train:
            x = augment_skeleton(x)

        return torch.FloatTensor(x), self.labels[idx]

    # 단독 Dataset 용 __getitem__
    def __getitem__(self, idx):
        data = self._get_data(idx)
        indices = self._sample_indices(len(data))
        return self.get_clip(idx, indices)

    # 내부에서 랜덤 clip 샘플링
    def _sample_indices(self, num_frames, clip_len=30):
        if num_frames < clip_len:
            return list(range(num_frames)) + [num_frames - 1] * (clip_len - num_frames)
        start = random.randint(0, num_frames - clip_len)
        return [start + i for i in range(clip_len)]