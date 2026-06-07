import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from src.model.attention_policy import AttentionPolicy


class PaperAttentionDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.samples = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        x = torch.tensor(item["features"], dtype=torch.float32)
        mask = torch.tensor(item["mask"], dtype=torch.bool)
        y = torch.tensor(item["target"], dtype=torch.long)

        return x, mask, y


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PaperAttentionDataset(
        "data/train_samples/paper_attention_cvrp20.pkl"
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    model = AttentionPolicy(input_dim=6, hidden_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 20

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, mask, y in loader:
            x = x.to(device)
            mask = mask.to(device)
            y = y.to(device)

            logits = model(x, mask)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        acc = correct / total * 100

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss = {total_loss / len(loader):.4f} | "
            f"Acc = {acc:.2f}%"
        )

    Path("results").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "results/paper_attention_cvrp20.pt")

    print("Saved model: results/paper_attention_cvrp20.pt")


if __name__ == "__main__":
    train()