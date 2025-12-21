import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.graphs import Graph, ST_GCN_Block
from .layers.transformers import DualFusionTransformer
from .layers.attentions import TemporalAttention
from .layers.encoders import RGB_Encoder, Skeleton_Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalDecisionHead(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=256, num_classes=5):
        super().__init__()
        
        # 1. 입력된 채널을 압축하여 시간(T) 축의 특징만 추출함
        self.gate_conv1 = nn.Conv1d(in_channels, in_channels // 8, 
                                    kernel_size=3, padding=1, bias=False)
        self.gate_relu = nn.ReLU(inplace=True)
        # 최종 점수를 1개(0~1)로 뽑아내는 레이어 (반드시 bias=True)
        self.gate_conv2 = nn.Conv1d(in_channels // 8, 1, 
                                    kernel_size=3, padding=1, bias=True) 
        self.gate_sigmoid = nn.Sigmoid()
        
        # ==== 초기 학습 시 모든 프레임을 통과시키기 위한 bias 초기화 ====
        if self.gate_conv2.bias is not None:
            nn.init.constant_(self.gate_conv2.bias, 1.0)

        # 2. 백엔드 로직
        self.backend_conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Input x: (B, C, T, V) - Fusion과 Mixer를 거친 시공간 특징
        """
        B, C, T, V = x.shape
        
        # 1. 공간 압축 (기존 TemporalAttention의 y = F.avg_pool2d 로직)
        # 모든 관절(V)의 정보를 평균내어 '프레임별 대표값' 생성
        y = x.mean(dim=-1) # (B, C, T)

        # 2. 중요도(Attention Score) 계산 (기존 score = self.fc(y) 로직)
        g = self.gate_relu(self.gate_conv1(y))
        scores = self.gate_sigmoid(self.gate_conv2(g)) # (B, 1, T)
        
        # 3. 중요도 적용 (게이팅)
        # 중요한 프레임은 강조하고, 잡음 프레임은 억제합니다. 
        # (0.5를 더해주는 것은 학습 초기 정보 소실을 방지하는 보험입니다.)
        x_weighted = y * (0.1 + scores) 
        
        # 4. 최종 분류 경로
        x_feat = self.backend_conv(x_weighted)      # 시간적 문맥 파악 (B, 256, T)
        x_flat = self.global_pool(x_feat).squeeze(-1) # 시간 축 압축 (B, 256)
        
        logits = self.fc(x_flat) # 최종 분류 (B, num_classes)
        
        return logits, scores # 분석을 위해 점수(scores)도 함께 리턴합니다.
        
class SGCH_Net(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.graph = Graph()
        A = self.graph.A
        self.rgb_stream = RGB_Encoder()
        self.skel_stream = Skeleton_Encoder(3, 128, A)
        self.fusion = DualFusionTransformer(d_model=128, num_queries=21, nhead=4, num_layers=2)
        self.temporal = TemporalAttention(d_model=128) # 관절별 시간 흐름 파악
        self.head = TemporalDecisionHead(
            in_channels=128, 
            hidden_dim=256, 
            num_classes=num_classes
        )

    def forward(self, x_rgb, x_skel):
        # x_rgb: (B, 3, T, H, W)
        # x_skel: (B, T, V, C)
        
        B, T, V, C = x_skel.size()

        # # 손목 원점 이동 및 크기 정규화
        # wrist = x_skel[:, :, 0:1, :] 
        # x_skel = x_skel - wrist
        
        # 시퀀스 전체에서 가장 큰 값을 찾아 하나로 통일해서 정규화
        max_dist_seq = torch.norm(x_skel, dim=3).max(dim=2).values.max(dim=1, keepdim=True).values
        x_skel = x_skel / (max_dist_seq[:, :, None, None] + 1e-6)
        
        # ST-GCN 입력 포맷으로 변경: (B, C, T, V)
        x_skel = x_skel.permute(0, 3, 1, 2).contiguous() 
        
        feat_rgb = self.rgb_stream(x_rgb)            # (B, 128, T, 7, 7)
        feat_skel, skel_attn = self.skel_stream(x_skel) # (B, 128, T, V) + Hybrid Attn
        
        feat_fused, fusion_attns = self.fusion(feat_skel, feat_rgb) 
        
        # (B*T, V, 128) -> (B, T, V, 128) -> (B, 128, T, V)
        x = feat_fused.view(B, T, V, -1).permute(0, 3, 1, 2).contiguous()
        x = self.temporal(x) # (B, 128, T, V)
    
        logits, temporal_scores = self.head(x) # 중요 프레임 선별 및 최종 분류
        
        analysis_data = {
            'skel_attn': skel_attn,       # SKe Encoder의 그래프 어텐션
            'fusion_attns': fusion_attns, # Transformer의 공간 융합 어텐션
            'temporal_scores': temporal_scores # 백엔드의 프레임 점수
        }
        
        return logits, analysis_data