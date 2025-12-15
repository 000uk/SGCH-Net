import random
import math
import numpy as np
from torch.utils.data import Dataset
import torch

# ==========================================
# 2. 공간 증강 함수 (Spatial Augmentation)
# ==========================================
def augment_skeleton(x, label=None):
    """
    x shape: (30, 21, 3) -> (프레임, 관절, 좌표)
    이미 손목이 (0,0,0)으로 정규화된 상태에서 들어와야 회전이 자연스러움
    """

    # 1. Random Rotation (회전)
    # 50% 확률로 -30도 ~ +30도 (라디안 -0.5 ~ 0.5) 회전
    if random.random() < 0.5:
        theta = random.uniform(-0.5, 0.5) # 영상 하나당 각도 1개만 뽑음!
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        
        # x, y 좌표 회전 (Z축 기준)
        x_new = x[:, :, 0] * cos_t - x[:, :, 1] * sin_t
        y_new = x[:, :, 0] * sin_t + x[:, :, 1] * cos_t
        
        x[:, :, 0] = x_new
        x[:, :, 1] = y_new

    # 2. Random Scaling (크기)
    # 50% 확률로 0.9 ~ 1.1배 크기 조절
    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1) 
        x = x * scale 

    # 3. Gaussian Noise (노이즈)
    # 50% 확률로 센서 노이즈 추가 (프레임/관절마다 다르게)
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.01, x.shape) 
        x = x + noise 
        
    return x

# ==========================================
# 1. 보간 함수 (Interpolation) - 필수!
# ==========================================
def interpolate_video(x, target_len=30):
    """
    x: (Current_Frames, 21, 3) -> 길이가 몇 개든 들어오면
    target_len: 30개로 부드럽게 늘리거나 줄여줌
    """
    curr_len, V, C = x.shape
    if curr_len == target_len:
        return x

    # 현재 길이에서 30개 지점을 균등하게 찍음 (소수점 인덱스)
    indices = np.linspace(0, curr_len - 1, target_len)
    
    target_x = np.zeros((target_len, V, C))
    for i, idx in enumerate(indices):
        lower = int(np.floor(idx))
        upper = int(np.ceil(idx))
        alpha = idx - lower
        
        # 선형 보간 (Linear Interpolation)
        target_x[i] = (1 - alpha) * x[lower] + alpha * x[upper]
        
    return target_x
    
# 1. 커스텀 데이터셋 정의
class SkeletonDataset(Dataset):
    def __init__(self, data_source, labels, num_frames, train=True):
        self.data = data_source
        self.labels = labels
        self.train = train
        self.fixed_len = num_frames
        
    def __len__(self):
        return len(self.data)
       
    def __getitem__(self, idx):
        # 1. 로드 (37, 21, 3)
        path = self.data[idx]
        x = np.load(path)
        
        # ============================================
        # [Augmentation 1] 시간 축 흔들기 (Resampling)
        # ============================================
        if self.train:
            # 37프레임 중 랜덤하게 25~37개만 선택
            crop_len = random.randint(25, self.fixed_len)

            # 전체 구간에서 랜덤하게 n개 뽑기 (순서 유지; sorted)
            # indices = sorted(random.sample(range(self.fixed_len), crop_len))
            # x = x[indices]   # 예: (28, 21, 3) 마치 동작을 좀 더 빠르게 한 것처럼 만듦

            # 얜 전체구간 ㄴㄴ
            start = random.randint(0, self.fixed_len - crop_len)
            x = x[start:start + crop_len]

        # 다시 37프레임으로 복구(보간법)
        # 예) 24개짜리를 30개로 늘리니까 동작이 부드럽게 이어짐
        x = interpolate_video(x, target_len=self.fixed_len)
        
        # ============================================
        # [Augmentation 2] 프레임 드랍 (Frame Dropout)
        # ============================================
        if self.train and random.random() < 0.5:
            # 30개 중 랜덤하게 1~5개 프레임을 0으로 만듦 (가려짐 효과)
            drop_indices = random.sample(range(self.fixed_len), k=random.randint(1, 5))
            x[drop_indices] = 0

        # 3. 좌표 정규화 (손목 중심)
        # 공간 증강(회전) 전에 반드시 손목을 (0,0,0)으로 맞춰야 제자리에서 돕니다.
        wrist = x[:, 0:1, :]
        x = x - wrist
        
        # ============================================
        # [Augmentation 3] 공간 증강 (회전/스케일/노이즈)
        # ============================================
        if self.train:
            x = augment_skeleton(x, self.labels[idx])

        return torch.FloatTensor(x), torch.tensor(self.labels[idx], dtype=torch.long)