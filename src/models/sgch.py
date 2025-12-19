import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.graph_layers import Graph, TemporalAttention, ST_GCN_Block
from .layers.transformer_blocks import CMFA_Fusion, DualFusionTransformer
from torchvision.models import mobilenet_v3_small

class RGB_Encoder(nn.Module): # ★ 2D CNN (MobileNet)
    def __init__(self):
        super(RGB_Encoder, self).__init__()
        self.backbone = mobilenet_v3_small(pretrained=True).features
        self.conv_project = nn.Conv2d(576, 128, kernel_size=1) 

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        x = self.backbone(x)     # (B*T, 576, 7, 7)
        x = self.conv_project(x) # (B*T, 128, 7, 7)
        _, C_new, H_new, W_new = x.shape
        x = x.view(B, T, C_new, H_new, W_new).permute(0, 2, 1, 3, 4) 
        return x # (B, 128, T, 7, 7)

class Skeleton_Encoder(nn.Module): # ★ ST-GCN Part
    def __init__(self, in_channels, out_channels, A):
        super(Skeleton_Encoder, self).__init__()
        self.gcn_networks = nn.Sequential(
            ST_GCN_Block(in_channels, 64, A),
            ST_GCN_Block(64, 64, A),
            ST_GCN_Block(64, 128, A) # 128채널로 맞춰줌
        )
    def forward(self, x):
        for gcn in self.gcn_networks: x = gcn(x)
        return x

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
        feat_skel = self.skel_stream(x_skel)
        
        # Fusion (Skeleton Guides RGB)
        x, attn_map = self.fusion(feat_skel, feat_rgb)
        
        # Backend Processing
        for block in self.backend: x = block(x)
        
        # 중요하지 않은 프레임(준비/원위치 동작)을 마스킹
        x = self.temporal_attn(x)
        
        # Classification
        x = F.avg_pool2d(x, x.size()[2:]).view(N, -1)
        return self.fc(x), attn_map