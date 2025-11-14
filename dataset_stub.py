import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np

class SimpleTrafficDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.files = [os.path.join(img_dir,f) for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        im = cv2.imread(path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (640,640))
        im = im.astype(np.float32) / 255.0
        im_t = torch.from_numpy(im).permute(2,0,1).contiguous()
        annotations = {
            "boxes": torch.tensor([[50,50,200,200]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.long)
        }
        density = torch.zeros((1,640,640), dtype=torch.float32)
        return im_t, annotations, density
