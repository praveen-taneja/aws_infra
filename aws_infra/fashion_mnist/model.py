# https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
import os

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

    return training_data, test_data, train_data_loader, test_data_loader

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

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X,y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss = test_loss + loss_fn(pred, y).item()
        test_loss = test_loss/num_batches
        print(f"Test avg loss = {test_loss}")

def model_predict(model, x):

    #predict
    model.eval()
    #x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        return CLASSES[pred[0].argmax(0)]
        #predicted, actual = CLASSES[pred[0].argmax(0)], CLASSES[y]
        #print(f"Predicted = {predicted}, Actual = {actual}")

if __name__ == "__main__":

    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 5
    CLASSES = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    parent_dir = os.path.join(os.path.dirname(__file__))

    training_data, test_data, train_data_loader, test_data_loader = load_data(
        batch_size=BATCH_SIZE)
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
        test(test_data_loader, model, loss_fn)
        print("Done!")

    save_model_path = os.path.join(parent_dir, "model.pth")
    torch.save(model.state_dict(), save_model_path)
    print(f"Saved model to {save_model_path}")

    #load model
    model = FashionModel()
    model.load_state_dict(torch.load(save_model_path))

    #predict
    x, y = test_data[0][0], test_data[0][1]
    predicted = model_predict(model, x)
    actual = CLASSES[y]
    print(f"predicted = {predicted}")
    print(f"actual = {actual}")