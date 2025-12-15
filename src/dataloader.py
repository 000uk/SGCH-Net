import pandas as pd
from datasets import DataLoader
from sklearn.model_selection import train_test_split

from .datasets.rgb import VideoDataset
from .datasets.skeleton import SkeletonDataset
from .datasets.multimodal import MultiModalDataset

def get_loader(BASE_PATH, batch_size=8, num_workers=4,
               use_rgb=True, use_skel=True):

    df = pd.read_csv(f"{BASE_PATH}/meta.csv")
    labels = df["label_id"].tolist()

    video_paths = [
        f"{BASE_PATH}/frames/{lbl}/{vid}"
        for lbl, vid in zip(df["label"], df["video_id"])
    ]
    npy_paths = [
        f"{BASE_PATH}/mediapipes/{lbl}/{vid}.npy"
        for lbl, vid in zip(df["label"], df["video_id"])
    ]

    vid_tr, vid_val, npy_tr, npy_val, y_tr, y_val = train_test_split(
        video_paths, npy_paths, labels,
        test_size=0.2, stratify=labels, random_state=42
    )

    rgb_tr = VideoDataset(vid_tr, y_tr, train=True) if use_rgb else None
    rgb_val = VideoDataset(vid_val, y_val, train=False) if use_rgb else None

    skel_tr = SkeletonDataset(npy_tr, y_tr, train=True) if use_skel else None
    skel_val = SkeletonDataset(npy_val, y_val, train=False) if use_skel else None

    train_ds = MultiModalDataset(rgb_tr, skel_tr)
    val_ds   = MultiModalDataset(rgb_val, skel_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)

    return train_loader, val_loader