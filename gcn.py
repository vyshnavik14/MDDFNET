import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        Ax = torch.matmul(adj, x)
        out = self.linear(Ax) + self.bias
        return F.relu(out)

class GCNFusion(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, out_dim=128, num_layers=2):
        super().__init__()
        layers = []
        layers.append(SimpleGCNLayer(in_dim, hidden_dim))
        for _ in range(max(0, num_layers-2)):
            layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))
        layers.append(SimpleGCNLayer(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, out_dim)
        )

    def forward(self, node_feats, adj):
        h = node_feats
        for l in self.layers:
            h = l(h, adj)
        fused = self.classifier(h)
        graph_embedding = fused.mean(dim=0)
        return fused, graph_embedding
