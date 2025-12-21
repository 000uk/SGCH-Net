import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .attentions import CoordAtt

class Graph:
    def __init__(self, strategy='uniform'):
        self.num_node = 21
        self.edges = self.get_edges()
        self.A = self.get_adjacency_matrix(self.edges, self.num_node)

    def get_edges(self):
        # MediaPipe Hands 기준 연결 정보
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_link = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]
        return self_link + neighbor_link

    def get_adjacency_matrix(self, edges, num_node):
        A = np.zeros((num_node, num_node))
        for i, j in edges:
            A[j, i] = 1
            A[i, j] = 1
        DL = np.sum(A, 0)
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if DL[i] > 0:
                Dn[i, i] = DL[i]**(-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)
        return torch.tensor(DAD, dtype=torch.float32)

class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.A = nn.Parameter(A, requires_grad=False)

    def forward(self, x):
        x = self.conv(x)
        n, c, t, v = x.size()
        x = x.view(n, c * t, v)
        x = torch.matmul(x, self.A) 
        x = x.view(n, c, t, v)
        return x
    
class ST_GCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1):
        super(ST_GCN_Block, self).__init__()

        self.gcn = GraphConv(in_channels, out_channels, A)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (9, 1), # (Time, Node) Node는 섞지 않고 시간축으로 9프레임을 봄
                padding=(4, 0),
                stride=(stride, 1)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5, inplace=True)
        )
        
        self.attention = CoordAtt(out_channels) 

        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = self.attention(x)
        return F.relu(x + res)