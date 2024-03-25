# Kevin Heleodoro - MNIST digit recognition using CNN

# ------ Import Statements ------  #

import argparse  # Argument parser
import torch  # PyTorch
import torchvision  # Vision module
import torch.nn as nn  # Neural network module
import torch.nn.functional as F  # Functional module
import torch.optim as optim  # Optimizer
import matplotlib.pyplot as plt  # Plotting
from torch.utils.data import DataLoader  # Data loader
from torchvision import datasets  # Datasets
from torchvision.transforms import ToTensor  # Transform to tensor
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

# ------ Global Variables ------  #

learning_rate = 0.01
momentum = 0.5
n_epochs = 5

# ------ Class Definitions ------  #


# network for MNIST digit recognition
class MyNetwork(nn.Module):
    # Constructor
    def __init__(self):
        super().__init__()
        # Convolutional layer (10 5x5 filters)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # Convolutional layer (20 5x5 filters)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Dropout layer (50%)
        self.dropout = nn.Dropout(p=0.5)
        # Fully connected layer (50 nodes)
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        # Final fully connected layer (10 nodes)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    # Forward pass for network
    def forward(self, x):
        # First convolutional layer, max pooling (2x2 window) and ReLU activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Second convolutional layer, dropout, max pooling (2x2 window) and ReLU activation
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        # Flatten tensor
        x = x.view(-1, 320)
        # x = x.view(-1, self.num_flat_features(x))
        # Apply ReLU activation to first fully connected layer
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer
        x = self.fc2(x)
        # Apply log softmax to output
        return F.log_softmax(x, dim=1)


# ------ Function Definitions ------  #


# Print a border
def print_border():
    print("\n" + "-" * 50 + "\n")


# Parse through the arguments
def arg_parser():
    # Creates the parser
    parser = argparse.ArgumentParser(description="MNIST digit recognition using CNN")

    # Download the MNIST dataset
    parser.add_argument(
        "--download", type=bool, default=True, help="Download the MNIST dataset"
    )

    # Save examples from the test data
    parser.add_argument(
        "--save_example",
        type=bool,
        default=False,
        help="Save first 6 examples from the test data",
    )

    # Creates visualizer for the network
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="Visualize the network using TensorBoard",
    )

    # Train the network
    parser.add_argument(
        "--train_network",
        type=bool,
        default=True,
        help="Train the network using the MNIST dataset",
    )

    # Parse the arguments
    return parser.parse_args()


# Download and load the MNIST dataset
def load_data():
    print_border()
    print("Downloading data...")

    # Parameters
    batch_size = 64
    mnist_mean = 0.1307
    mnist_standard_deviation = 0.3081

    # Training MNIST dataset
    training_data = datasets.MNIST(
        root="data/main",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (mnist_mean,), (mnist_standard_deviation,)
                ),
            ]
        ),
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    # Testing MNIST dataset
    test_data = datasets.MNIST(
        root="data/main",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (mnist_mean,), (mnist_standard_deviation,)
                ),
            ]
        ),
    )
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    print("Data loaded successfully")

    return train_dataloader, test_dataloader


# Save examples from the test data
def save_examples(test_dataloader):
    print_border()
    print("Saving examples from the test data...")
    examples = enumerate(test_dataloader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(f"Example data shape: {example_data.shape}")

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title(f"Ground Truth: {example_targets[i]}")
        plt.xticks([])
        plt.yticks([])
    file_path = "results/main/mnist_example_test_images.png"
    fig.savefig(file_path)
    print(f"Examples saved successfully: {file_path}")


# Train the network
def train_network(
    train_dataloader, network, optimizer, train_losses, train_counter, epoch
):
    print_border()
    print("Training the network...")

    # Set the network to training mode
    network.train()

    log_interval = 100

    # Loop through the epochs
    # for epoch in range(1, n_epochs + 1):
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()  # Zero the gradients
        output = network(data)  # Forward pass
        loss = F.nll_loss(output, target)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update the weights

        if batch_idx % log_interval == 0:
            print(
                f"Training Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} ({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_dataloader.dataset))
            )
            torch.save(network.state_dict(), "results/main/mnist_model.pth")
            torch.save(optimizer.state_dict(), "results/main/mnist_optimizer.pth")

    print("Training complete")


# Test the network
def test_network(test_dataloader, network, test_losses):
    print_border()
    print("Testing the network...")

    # Set the network to evaluation mode
    network.eval()

    test_loss = 0
    correct = 0

    # No gradient calculation
    with torch.no_grad():
        for data, target in test_dataloader:
            output = network(data)  # Forward pass
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # Calculate the loss
            pred = output.data.max(1, keepdim=True)[
                1
            ]  # Get the index of the max log-probability
            correct += pred.eq(
                target.data.view_as(pred)
            ).sum()  # Calculate the number of correct predictions

    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)
    print(
        f"\nTest set: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} ({100. * correct / len(test_dataloader.dataset):.2f}%)\n"
    )


# ------ Main Code ------  #


def main():
    print_border()
    print("\nMNIST digit recognition using CNN\n")

    # Parse through the arguments
    args = arg_parser()

    # Load the data
    if args.download:
        train_dataloader, test_dataloader = load_data()

    # Save examples from the test data
    if args.save_example:
        save_examples(test_dataloader)

    # Instantiate the network
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    print_border()
    print("Network architecture:")
    print(network)

    # Visualize the network using TensorBoard
    if args.visualize:
        writer = SummaryWriter("results/main")
        dataiter = iter(train_dataloader)
        images, labels = next(dataiter)
        writer.add_graph(network, images)
        writer.close()

    # Initialize the training and testing losses and counter
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_dataloader.dataset) for i in range(n_epochs + 1)]

    # Test the network pre-training
    print(f"Running a pre-training test...")
    test_network(test_dataloader, network, test_losses)

    # Train and test the network
    if args.train_network:
        for epoch in range(1, n_epochs + 1):
            train_network(
                train_dataloader,
                network,
                optimizer,
                train_losses,
                train_counter,
                epoch,
            )
            test_network(test_dataloader, network, test_losses)

    return


if __name__ == "__main__":
    main()
