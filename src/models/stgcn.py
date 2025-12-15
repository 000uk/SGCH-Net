import torch.nn as nn
import torch.nn.functional as F
from .layers.graph_layer.py import GraphConv, Graph
from .layers.attn_layer.py import CoordAtt, TemporalAttention

# Spatial(공간) → Temporal(시간) → Attention(중요도 필터링) → Residual(정보 보존)
"""
x = self.gcn(x): 먼저 손의 **모양(Spatial)**을 봅니다. (주먹인지, 펴고 있는지)
x = self.tcn(x): 그다음 손의 **움직임(Temporal)**을 봅니다. (멈춰있는지, 왼쪽으로 가는지)
x = self.attention(x) (핵심!):
앞서 추출한 모양과 움직임 정보 중에서 **"제스쳐 판단에 도움이 되는 채널"**은 증폭하고,
**"배경 노이즈나 쓸데없는 흔들림(Doing_Other_Things)"**과 관련된 채널은 억제합니다.
F.relu(x + res): 필터링된 정보에 원본 정보를 더해서(Residual) 다음 층으로 넘깁니다
"""
class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super(ST_GCN_Block, self).__init__()

        # Spatial Graph Conv
        self.gcn = GraphConv(in_channels, out_channels, A)

        # Temporal Conv (시간축으로 1D Conv)
        # kernel_size=(9, 1): 시간축으로 9프레임을 봄
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (9, 1), # (Time, Node) -> Node는 섞지 않고 시간만 봄
                padding=(4, 0),
                stride=(stride, 1)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5, inplace=True) # 과적합 방지
        )
        
        # Coordinate Attention 적용 (Spatial)
        self.attention = CoordAtt(out_channels) 

        # Residual Connection (ResNet처럼 입력 더하기)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x) # 공간
        x = self.tcn(x) # 시간
        x = self.attention(x) # Attention
        return F.relu(x + res)


class STGCN_Model(nn.Module):
    def __init__(self, num_classes=3, in_channels=3, num_frames=30):
        super(STGCN_Model, self).__init__()

        # 그래프 구조 로드
        self.graph = Graph()
        A = self.graph.A

        # 레이어 쌓기 (채널 수를 점점 늘림: 3 -> 64 -> 128 -> 256)
        self.data_bn = nn.BatchNorm1d(in_channels * 21) # 입력 정규화

        self.st_gcn_networks = nn.ModuleList([
            ST_GCN_Block(in_channels, 64, A),
            ST_GCN_Block(64, 64, A),
            ST_GCN_Block(64, 128, A),
            ST_GCN_Block(128, 128, A),
            ST_GCN_Block(128, 256, A)
        ])

        # 마지막 블록의 채널수가 256이었으므로 channel=256
        self.temp_attention = TemporalAttention(channel=256, t_dim=num_frames)

        # Global Pooling & Classification
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input shape: (Batch, Frames, Nodes, Channels) -> (B, 30, 21, 3)
        # ST-GCN requires: (Batch, Channels, Frames, Nodes) -> (B, 3, 30, 21)
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()

        # Data Normalization
        x = x.view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, T, V)

        # Forward through ST-GCN blocks
        for gcn in self.st_gcn_networks:
            x = gcn(x)

        # Global Temporal Attention
        # 이제 모델이 알아서 "앞뒤 의미 없는 프레임"의 값을 0으로 만들어버림
        x = self.temp_attention(x)
        
        # Global Average Pooling
        # 시간(Time)과 관절(Node) 차원을 모두 평균냄 -> "영상 전체의 특징 벡터"
        x = F.avg_pool2d(x, x.size()[2:]) # (N, 256, 1, 1)
        x = x.view(N, -1) # (N, 256)

        # Prediction
        x = self.fc(x)
        return x