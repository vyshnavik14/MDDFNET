import torch
from torch.utils.data import DataLoader
from models.zerosign import ZeroSign
from data.dataset_stub import SimpleTrafficDataset
import torch.optim as optim
from tqdm import tqdm

def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    ann = [b[1] for b in batch]
    den = torch.stack([b[2] for b in batch], dim=0)
    return imgs, ann, den

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ZeroSign(num_classes=43, pretrained=True).to(device)

    dataset = SimpleTrafficDataset("data/images")
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    for epoch in range(1,6):
        model.train()
        pbar = tqdm(loader)
        for imgs, ann, density in pbar:
            imgs = imgs.to(device)
            out = model(imgs)
            pred_density = out["density"]
            loss = pred_density.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Epoch {epoch} Loss {loss.item():.4f}")
    torch.save(model.state_dict(), "zerosign_checkpoint.pth")

if __name__ == "__main__":
    main()
