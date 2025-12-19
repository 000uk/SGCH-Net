import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from .graphs import ST_GCN_Block
from .attentions import HybridAttentionLayer

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

# class Skeleton_Encoder(nn.Module): # ★ ST-GCN Part
#     def __init__(self, in_channels, out_channels, A):
#         super(Skeleton_Encoder, self).__init__()
#         self.gcn_networks = nn.Sequential(
#             ST_GCN_Block(in_channels, 64, A),
#             ST_GCN_Block(64, 64, A),
#             ST_GCN_Block(64, 128, A) # 128채널로 맞춰줌
#         )
#     def forward(self, x):
#         for gcn in self.gcn_networks: x = gcn(x)
#         return x
class Skeleton_Encoder(nn.Module):
    def __init__(self, in_channels, d_model, A):
        super().__init__()
        # A는 (21, 21) 형태의 인접 행렬
        self.joint_encoder = LearnedJointEncoding(d_model)
        self.hybrid_attn = HybridAttentionLayer(d_model, nhead=4)
        
        # [수정] zeros가 아니라 실제 그래프 구조 A로 초기화!
        # 아주 작은 값(1e-4)이라도 줘서 연결된 곳에 특혜를 주자.
        self.graph_bias = nn.Parameter(A.clone().float()) 

        self.stgcn1 = ST_GCN_Block(d_model, d_model, A)
        self.stgcn2 = ST_GCN_Block(d_model, d_model, A)

    def forward(self, x):
        # x: (B, 3, T, V) -> 입력 포맷 확실하게!!!
        B, C, T, V = x.shape

        # 1) Coordinate Injection을 위해 차원 변경
        # (B, 3, T, V) -> (B, T, V, 3) -> (B*T, V, 3)
        coords = x.permute(0, 2, 3, 1).contiguous().view(B*T, V, C)

        # 2) 관절 의미 부여 (Joint Identity + Coord)
        feat = self.joint_encoder(coords) # (B*T, V, d_model)

        # 3) Hybrid Attention (전역 정보 + 그래프 위상 가이드)
        # 여기서 graph_bias가 바코드 현상을 막아줄 거임
        feat, attn_map = self.hybrid_attn(
            feat, feat, feat,
            graph_bias=self.graph_bias
        )

        # 4) ST-GCN 처리를 위해 원래 포맷으로 복구
        # (B*T, V, d_model) -> (B, T, V, d_model) -> (B, d_model, T, V)
        feat = feat.view(B, T, V, -1).permute(0, 3, 1, 2).contiguous()
        
        feat = self.stgcn1(feat)
        feat = self.stgcn2(feat)

        return feat, attn_map # 히트맵 확인을 위해 attn_map도 리턴해!

class LearnedJointEncoding(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        # 실제 (x, y, z) 좌표를 d_model 차원으로 뻥튀기
        self.coord_proj = nn.Linear(3, d_model)
        # 관절 ID (0~20) 마다 고유한 특징 부여
        self.joint_embed = nn.Embedding(21, d_model)
        
    def forward(self, coords):
        # coords: (B*T, 21, 3)
        joint_ids = torch.arange(21, device=coords.device)
        # 물리 좌표 정보 + 관절 고유 정보 결합
        return self.coord_proj(coords) + self.joint_embed(joint_ids)