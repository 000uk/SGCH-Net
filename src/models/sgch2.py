import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.graphs import Graph, ST_GCN_Block
from .layers.transformers import DualFusionTransformer
from .layers.attentions import TemporalAttention
from .layers.encoders import RGB_Encoder, Skeleton_Encoder

class SGCH_Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.graph = Graph()
        A = self.graph.A
        
        # 1. Encoders
        self.rgb_stream = RGB_Encoder()
        self.skel_stream = Skeleton_Encoder(3, 128, A)
        
        # 2. Fusion
        self.fusion = DualFusionTransformer(d_model=128, num_queries=21, nhead=4, num_layers=2)
        
        # 3. Backend (Deep Features)
        self.backend = nn.Sequential(
            ST_GCN_Block(128, 256, A, stride=2),
            ST_GCN_Block(256, 256, A)
        )
        
        # 4. Global Temporal Attention
        self.temporal_attn = TemporalAttention(256)
        
        # 5. Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x_rgb, x_skel):
        # x_rgb: (B, 3, T, H, W)
        # x_skel: (B, T, V, C) -> (B, C, T, V) 일케 바꿀건데 정규화도 할 예정

        # ======================================
        # Transformer 학습을 위한 정규화 (Normalization)
        # 이게 왜 되냐면... 지금 global attention 방식이라서....!
        # 어차피 트랜스포머는 MatMul이랑 Softmax 쓰니까 정규화 안하면 좌표값 클때 계산 터져버림
        # self.pos_embed_rgb 임마 덕에 ske이랑 rgb랑 매칭 됨
        # 1. 손목 원점 이동
        wrist = x_skel[:, :, 0:1, :] # (B, T, 1, C) - 0번 관절이 손목이라고 가정
        x_skel = x_skel - wrist
        
        # 2. 크기 통일
        # (B, T, V, C) -> norm dim=3 (xyz) -> max dim=2 (joints)
        max_dist = torch.norm(x_skel, dim=3).max(dim=2).values # (B, T)
        max_dist[max_dist == 0] = 1e-6 
        x_skel = x_skel / max_dist[:, :, None, None]
        # ======================================
        
        N, T, V, C = x_skel.size()
        # (B, T, V, C) -> (B, C, T, V) 로 변환 (ST-GCN 입력 포맷)
        x_skel = x_skel.permute(0, 3, 1, 2).contiguous() 
        
        # Encoding
        feat_rgb = self.rgb_stream(x_rgb) 
        feat_skel, skel_attn = self.skel_stream(x_skel)
        
        # Fusion (Skeleton Guides RGB)
        x, attn_maps = self.fusion(feat_skel, feat_rgb)
        
        # Backend Processing
        for block in self.backend: x = block(x)
        
        # 중요하지 않은 프레임(준비/원위치 동작)을 마스킹
        x = self.temporal_attn(x)
        
        # Classification
        x = F.avg_pool2d(x, x.size()[2:]).view(N, -1)
        return self.fc(x), (skel_attn, *attn_maps)