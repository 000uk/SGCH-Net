from ..utils import sample_clip
from torch.utils.data import Dataset

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
        if self.rgb is not None:
            label = self.rgb.get_label(idx)
            T = len(self.rgb._get_frame_paths(idx))
        elif self.skel is not None:
            skel, label = self.skel[idx]
            T = skel.shape[0]
        else:
            raise ValueError

        indices = sample_clip(T, self.clip_len, self.min_stride, self.max_stride)

        out = {"label": int(label)}

        if self.rgb is not None:
            out["rgb"], _ = self.rgb.get_clip(idx, indices)

        if self.skel is not None:
            out["skel"], _ = self.skel.get_clip(idx, indices)

        return out