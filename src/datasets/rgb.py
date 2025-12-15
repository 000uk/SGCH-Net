from ..utils import sample_clip
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels,
                 clip_len=30, min_stride=2, max_stride=6, train=True):
        """
        data_list: [.npy 파일 경로 리스트] or [Numpy 배열 리스트]
        labels: [0, 1, 0, 2, ...] 정답 라벨 리스트
        """
        self.video_paths = video_paths
        self.labels = labels
        self.clip_len = clip_len
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.train = train

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.video_paths)

    # 프레임 경로 가져오기
    def _get_frame_paths(self, idx):
        video_dir = self.video_paths[idx]
        frames = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith('.jpg')
        ])
        return frames

    # 원랜 __getitem__에 있던건데 ske이랑 시간축 맞출려고 뺌
    def get_clip(self, idx, indices):
        frame_paths = self._get_frame_paths(idx)
        label = self.labels[idx]

        images = []
        for i in indices:
            i = min(i, len(frame_paths) - 1)
            try:
                img = Image.open(frame_paths[i]).convert("RGB")
                img = self.transform(img)
                images.append(img)
            except Exception:
                images.append(torch.zeros(3, 224, 224))

        # (T, C, H, W) -> (C, T, H, W)
        video_tensor = torch.stack(images).permute(1, 0, 2, 3)
        return video_tensor, label

    def __getitem__(self, idx):
        frames = self._get_frame_paths(idx)
        indices = sample_clip(
            len(frames),
            clip_len=self.clip_len,
            min_stride=self.min_stride,
            max_stride=self.max_stride
        )
        return self.get_clip(idx, indices)