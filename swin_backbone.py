import torch
import torch.nn as nn
import timm

class SwinBackbone(nn.Module):
    """Swin Transformer backbone wrapper returning multi-scale feature maps."""
    def __init__(self, model_name="swin_base_patch4_window7_224", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, features_only=True, out_indices=(0,1,2,3))
        feat_info = self.model.feature_info
        self.out_channels = [f['num_chs'] for f in feat_info]

    def forward(self, x):
        feats = self.model(x)
        return feats
