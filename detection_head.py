import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    """Simple multi-scale detection head (prototype)."""
    def __init__(self, in_channels_list, num_classes=43, hidden_dim=256):
        super().__init__()
        self.stages = nn.ModuleList()
        for in_ch in in_channels_list:
            seq = nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
            self.stages.append(seq)
        self.reg_heads = nn.ModuleList([nn.Conv2d(hidden_dim, 4, kernel_size=1) for _ in in_channels_list])
        self.obj_heads = nn.ModuleList([nn.Conv2d(hidden_dim, 1, kernel_size=1) for _ in in_channels_list])
        self.cls_heads = nn.ModuleList([nn.Conv2d(hidden_dim, num_classes, kernel_size=1) for _ in in_channels_list])

    def forward(self, feats):
        outputs = []
        for i, f in enumerate(feats):
            t = self.stages[i](f)
            reg = self.reg_heads[i](t)
            obj = self.obj_heads[i](t)
            cls = self.cls_heads[i](t)
            outputs.append({"reg": reg, "obj": obj, "cls": cls})
        return outputs
