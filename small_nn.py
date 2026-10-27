
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from adam import Adam

transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)


model = MLP()
criterion = nn.CrossEntropyLoss()

# Adam optimizer only sees and updates this flat NumPy vector
theta = np.concatenate([
    p.detach().cpu().numpy().ravel()
    for p in model.parameters()
]).copy()

optimizer = Adam(theta, lr=0.001)

def update_model_parameters(model, theta):
    offset = 0

    with torch.no_grad():
        for p in model.parameters():
            size = p.numel()

            values = theta[offset:offset + size]
            values = values.reshape(p.shape)

            p.copy_(
                torch.as_tensor(
                    values,
                    dtype=p.dtype,
                    device=p.device
                )
            )
            offset += size

    assert offset == len(theta)


def train(model, train_loader, optimizer, epochs=5):

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        total_samples = 0
        total_correct = 0

        for images, labels in train_loader:

            model.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            gradients = np.concatenate([
                p.grad.detach().cpu().numpy().ravel()
                for p in model.parameters()
            ])

            theta = optimizer.step(gradients)

            update_model_parameters(model, theta)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)
            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

        avg_loss = total_loss / total_samples
        accuracy = 100 * total_correct / total_samples

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Loss: {avg_loss} | "
            f"Accuracy: {accuracy}%"
        )


if __name__ == "__main__":
    train(model, train_loader, optimizer, epochs=5)
