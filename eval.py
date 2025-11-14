import torch
from models.zerosign import ZeroSign
from data.dataset_stub import SimpleTrafficDataset
from torch.utils.data import DataLoader
import numpy as np

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroSign(num_classes=43, pretrained=False).to(device)
    model.eval()
    dataset = SimpleTrafficDataset("data/images")
    loader = DataLoader(dataset, batch_size=1)
    with torch.no_grad():
        for imgs, ann, den in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            # placeholder: print graph embedding shape and density map mean
            print('Graph emb shape:', out['graph_embedding'].shape, 'Density mean:', out['density'].mean().item())

if __name__ == '__main__':
    evaluate()
