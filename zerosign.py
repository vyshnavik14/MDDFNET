import torch
import torch.nn as nn
import torch.nn.functional as F
from .swin_backbone import SwinBackbone
from .detection_head import DetectionHead
from .density_head import DensityHead
from .gcn import GCNFusion

class ZeroSign(nn.Module):
    def __init__(self, num_classes=43, swin_name="swin_base_patch4_window7_224", pretrained=True):
        super().__init__()
        self.backbone = SwinBackbone(model_name=swin_name, pretrained=pretrained)
        in_chs = self.backbone.out_channels
        use_chs = in_chs[:3]
        self.det_head = DetectionHead(use_chs, num_classes=num_classes, hidden_dim=256)
        self.density_head = DensityHead(in_channels=in_chs[-1], hidden=128)
        node_feat_dim = 256
        self.node_proj = nn.Linear(node_feat_dim, node_feat_dim)
        self.gcn = GCNFusion(in_dim=node_feat_dim, hidden_dim=128, out_dim=128, num_layers=3)

    def forward(self, x):
        feats = self.backbone(x)
        det_feats = feats[:3]
        det_preds = self.det_head(det_feats)
        density = self.density_head(feats[-1], upsample_size=x.shape[2:])

        node_list = []
        for df in det_feats:
            g = F.adaptive_avg_pool2d(df, (1,1)).flatten(1)
            node_proj = self.node_proj(g)
            node_list.append(node_proj)
        dens_pool = F.adaptive_avg_pool2d(feats[-1], (1,1)).flatten(1)
        dens_node = self.node_proj(dens_pool)
        node_list.append(dens_node)

        B = x.shape[0]
        fused_nodes_all = []
        graph_embs = []
        for b in range(B):
            nodes_b = torch.stack([n[b] for n in node_list], dim=0)
            N = nodes_b.size(0)
            adj = torch.ones((N,N), device=nodes_b.device)
            adj = adj / N
            fused, graph_emb = self.gcn(nodes_b, adj)
            fused_nodes_all.append(fused)
            graph_embs.append(graph_emb)
        out = {
            "detection": det_preds,
            "density": density,
            "graph_embedding": torch.stack(graph_embs, dim=0)
        }
        return out
