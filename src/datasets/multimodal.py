import random
from datasets import Dataset

def sample_clip(num_frames_total, clip_len=30, min_stride=2, max_stride=6):
    if num_frames_total < clip_len:
        return list(range(num_frames_total)) + \
               [num_frames_total - 1] * (clip_len - num_frames_total)

    stride = random.randint(min_stride, max_stride)
    max_start = max(0, num_frames_total - clip_len * stride)
    start = random.randint(0, max_start) if max_start > 0 else 0

    idxs = [start + i * stride for i in range(clip_len)]
    idxs = [min(i, num_frames_total - 1) for i in idxs]

    return idxs


class MultiModalDataset(Dataset):
    def __init__(self, rgb_dataset=None, skel_dataset=None,
                 clip_len=30, min_stride=2, max_stride=6):
        assert rgb_dataset or skel_dataset
        self.rgb = rgb_dataset
        self.skel = skel_dataset
        self.clip_len = clip_len
        self.min_stride = min_stride
        self.max_stride = max_stride

        self.length = len(rgb_dataset or skel_dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.rgb:
            frames, label = self.rgb.get_raw(idx)
            T = len(frames)
        else:
            skel, label = self.skel.get_raw(idx)
            T = skel.shape[0]

        indices = sample_clip(
            T,
            clip_len=self.clip_len,
            min_stride=self.min_stride,
            max_stride=self.max_stride
        )

        out = {"label": label}

        if self.rgb:
            out["rgb"] = self.rgb.get_clip(idx, indices)

        if self.skel:
            out["skel"] = self.skel.get_clip(idx, indices)

        return out
