import random
from datasets import Dataset
from PIL import Image
import torch
from torchvision import transforms

def sample_clip(num_frames_total, clip_len=16, min_stride=2, max_stride=6):
    """연속된 클립 샘플링 (stride 랜덤)"""
    if num_frames_total < clip_len:  # 프레임 부족 시
        return list(range(num_frames_total)) + [num_frames_total - 1] * (clip_len - num_frames_total)

    stride = random.randint(min_stride, max_stride)
    max_start = max(0, num_frames_total - clip_len * stride)

    start = random.randint(0, max_start) if max_start > 0 else 0
    idxs = [start + i * stride for i in range(clip_len)]

    # --- 수정 포인트: 범위 넘으면 마지막 프레임으로 대체 ---
    idxs = [min(i, num_frames_total - 1) for i in idxs]

    return idxs

class VideoDataset(Dataset):
    def __init__(self, video_paths, labels,
                 num_frames=30, min_stride=2, max_stride=6, train=True):
        """
        data_list: [.npy 파일 경로 리스트] or [Numpy 배열 리스트]
        labels: [0, 1, 0, 2, ...] 정답 라벨 리스트
        """
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
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

    def __getitem__(self, idx):
        video_dir = self.video_paths[idx] # 폴더 경로
        label = self.labels[idx]

        # 폴더 안의 jpg 목록 가져오기
        frames = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')])
        
        # 연속된 클립 샘플링
        indices = sample_clip(len(frames), clip_len=self.num_frames)
        
        selected_frames = [frames[i] for i in indices]

        # 3. 이미지 로드 및 전처리
        images = []
        for p in selected_frames:
            try:
                img = Image.open(p).convert('RGB')
                img = self.transform(img) # 전처리 적용
                images.append(img)
            except Exception as e:
                print(f"Error loading image {p}: {e}")
                # 에러나면 검은 화면 추가 (Shape 유지)
                images.append(torch.zeros(3, 224, 224))

        # (T, C, H, W) → (C, T, H, W)
        video_tensor = torch.stack(images).permute(1, 0, 2, 3)
        return video_tensor, label