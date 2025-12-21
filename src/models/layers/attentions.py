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
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        # 시간 축을 따라 어텐션을 수행 (Sequence Length = T)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x input: (B, C, T, V)
        B, C, T, V = x.shape
        
        # 시간 축 어텐션을 위해 차원 변경: (B*V, T, C)
        # 즉, 각 관절(V)마다 시간(T) 흐름을 따로 보겠다는 뜻!
        x = x.permute(0, 3, 2, 1).contiguous().view(B * V, T, C)
        
        # 1) Self-Attention on Time
        attn_out, _ = self.attn(x, x, x) #  t=1일 때의 손 모양과 t=10일 때의 손 모양이 어떤 상관관계가 있는지 계산
        x = self.norm(x + attn_out)
        
        # 2) Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # 다시 원래대로: (B, C, T, V)
        x = x.view(B, V, T, C).permute(0, 3, 2, 1).contiguous()
        return x

class GraphAwareAttention(nn.Module):
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
        
        # Linear Projection
        query = self.q_proj(q) # (B*T, V, C)
        key = self.k_proj(k)
        value = self.v_proj(v)
        
        # Multi-head split (B*T, nhead, V, head_dim)
        query = query.view(B_T, V, self.nhead, C // self.nhead).transpose(1, 2)
        key = key.view(B_T, V, self.nhead, C // self.nhead).transpose(1, 2)
        value = value.view(B_T, V, self.nhead, C // self.nhead).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scaling = (C // self.nhead) ** -0.5
        attn_score = torch.matmul(query, key.transpose(-2, -1)) * scaling
        
        # 핵심!!!!! Graph Bias 주입
        if graph_bias is not None:
            # # graph_bias를 (1, 1, V, V)로 만들어서 모든 배치/헤드에 더해줌
            # attn_score = attn_score + graph_bias.unsqueeze(0).unsqueeze(0)
            # bias가 너무 지배적이지 않도록 learnable parameter나 작은 상수를 곱함
            if not hasattr(self, 'graph_alpha'):
                self.graph_alpha = nn.Parameter(torch.tensor(0.01, device=q.device))
            attn_score = attn_score + (self.graph_alpha * graph_bias.unsqueeze(0).unsqueeze(0))
            
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)
        
        # 5. Output
        out = torch.matmul(attn_prob, value)
        out = out.transpose(1, 2).contiguous().view(B_T, V, C)
        return self.out_proj(out), attn_prob