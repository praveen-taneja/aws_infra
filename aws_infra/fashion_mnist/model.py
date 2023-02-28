# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def load_data(batch_size=64):
    training_data = datasets.FashionMNIST(root="data", train=True,
                                          transform=ToTensor(), download=True)
    test_data = datasets.FashionMNIST(root="data", train=False,
                                      transform=ToTensor(), download= True)
    train_data_loader = DataLoader(training_data, batch_size)
    test_data_loader = DataLoader(test_data, batch_size)

    return train_data_loader, test_data_loader

def view_data():
    for X, y in test_data_loader:
        print(f"X.shape = {X.shape}")
        print(f"y.shape = {y.shape}")
        plt.imshow(X[3].reshape((28, 28)))
        break

class FashionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    #size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss = {loss}")

if __name__ == "__main__":

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 5

    train_data_loader, test_data_loader = load_data(batch_size=BATCH_SIZE)
    # view_data()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device = {device}")

    model = FashionModel()
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for t in range(NUM_EPOCHS):
        print(f"training epoch = {t + 1}")
        train(train_data_loader, model, loss_fn, optimizer)
        print("Done!")