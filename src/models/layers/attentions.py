import torch
import torch.nn as nn
import torch.nn.functional as F

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()
        
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        n, c, t, v = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [t, v], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_h * a_w
    
class TemporalAttention(nn.Module):
    def __init__(self, channel, t_dim=30): # t_dim: 사용하는 프레임 수
        super(TemporalAttention, self).__init__()
        # (N, C, T, V) -> (N, 1, T, 1) : 시간 축만 살리고 다 압축
        self.avg_pool = nn.AdaptiveAvgPool2d((t_dim, 1))

        # 각 프레임의 중요도를 0~1 사이 점수로 매김
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // 8, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // 8, 1, kernel_size=1, bias=False), # 채널을 1개로 (점수판)
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (N, C, T, V)
        b, c, t, v = x.size()

        # 1. 전체 노드에 대해 평균을 내서 "이 시간대의 대표 값"을 뽑음
        y = F.avg_pool2d(x, (1, v)) # (N, C, T, 1)

        # 2. 어떤 시간이 중요한지 점수 계산
        score = self.fc(y) # (N, 1, T, 1) - 각 프레임 별 점수 (0~1)

        # 3. 중요도 곱하기 (중요한 프레임은 살리고, 잡음 프레임은 죽임)
        return x * score

class HybridAttentionLayer(nn.Module):
    def __init__(self, d_model=128, nhead=4, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, graph_bias=None):
        """
        graph_bias: (V, V) 형태의 인접 행렬 또는 학습 가능한 Bias
        """
        B_T, V, C = q.shape
        
        # 1. Linear Projection
        query = self.q_proj(q) # (B*T, V, C)
        key = self.k_proj(k)
        value = self.v_proj(v)
        
        # 2. Multi-head split (B*T, nhead, V, head_dim)
        query = query.view(B_T, V, self.nhead, C // self.nhead).transpose(1, 2)
        key = key.view(B_T, V, self.nhead, C // self.nhead).transpose(1, 2)
        value = value.view(B_T, V, self.nhead, C // self.nhead).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention
        scaling = (C // self.nhead) ** -0.5
        attn_score = torch.matmul(query, key.transpose(-2, -1)) * scaling
        
        # 4. [핵심] Graph Bias 주입
        if graph_bias is not None:
            # graph_bias를 (1, 1, V, V)로 만들어서 모든 배치/헤드에 더해줌
            attn_score = attn_score + graph_bias.unsqueeze(0).unsqueeze(0)
            
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)
        
        # 5. Output
        out = torch.matmul(attn_prob, value)
        out = out.transpose(1, 2).contiguous().view(B_T, V, C)
        return self.out_proj(out), attn_prob