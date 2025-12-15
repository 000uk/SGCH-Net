import os
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels, train=True):
        self.video_paths = video_paths
        self.labels = labels
        self.train = train

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.video_paths)

    # 🔥 전체 프레임 반환 (sampling X)
    def get_raw(self, idx):
        video_dir = self.video_paths[idx]
        label = self.labels[idx]
        frames = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith(".jpg")
        ])
        return frames, label

    # 🔥 sampling 결과로 tensor 생성
    def get_clip(self, idx, indices):
        frames, _ = self.get_raw(idx)
        images = []

        for i in indices:
            img = Image.open(frames[i]).convert("RGB")
            images.append(self.transform(img))

        video = torch.stack(images).permute(1, 0, 2, 3)  # (C,T,H,W)
        return video