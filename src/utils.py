import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)            # 기본 Python random 고정
    np.random.seed(seed)         # NumPy 랜덤 고정
    torch.manual_seed(seed)      # CPU 연산 랜덤 고정
    torch.cuda.manual_seed(seed) # GPU 모든 디바이스 랜덤 고정
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU일 때

    # 연산 재현성
    torch.backends.cudnn.deterministic = True  # cuDNN 연산을 determinisitc으로 강제
    torch.backends.cudnn.benchmark = False     # CUDA 성능 자동 튜닝 기능 끔 → 완전 재현 가능

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