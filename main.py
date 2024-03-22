# Kevin Heleodoro - MNIST digit recognition using CNN

#  ------ Import Statements ------  #
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#  ------ Global Variables ------  #

# Get cpu, gpu, or mps device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


#  ------ Class Definitions ------  #


# Define model
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # forward pass
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


#  ------ Function Definitions ------  #


# Use the training data to train the model
def train_model(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    return


# Download the dataset and return the DataLoaders
def download_data():
    # Training data
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    # Test data
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    batch_size = 64

    # Create data loaders which wraps an iterable over the dataset
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # Print out shape of the test data
    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader


# Evaluate the model's performance
def test_model(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# Save the state of the model locally
def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    print(f"Saved PyTorch Model State to {path}")


# Load the model from a local file
def load_model(path="model.pth"):
    model = MyNetwork().to(device)
    model.load_state_dict(torch.load(path))
    return model


# ------ Main Function ------  #


def main(argv):
    # Parse arguments

    # Load data
    train_dataloader, test_dataloader = download_data()

    # Initiate model
    model = MyNetwork().to(device)
    print(f"Using {device} device")
    print(model)

    # Loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # Train data over N epochs
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_model(train_dataloader, model, loss_fn, optimizer)
        test_model(test_dataloader, model, loss_fn)

    print("Done!")

    save_model(model, "model.pth")

    # main function code
    return


if __name__ == "__main__":
    main(sys.argv)
