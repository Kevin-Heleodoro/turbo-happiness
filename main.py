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

# ------ Global Variables ------  #


# ------ Class Definitions ------  #


# Model for MNIST digit recognition
class MyNetwork(nn.Module):
    # Constructor
    def __init__(self):
        pass

    # Forward pass for network
    def forward(self, x):
        pass


# ------ Function Definitions ------  #
# Print a border
def print_border():
    print("\n" + "-" * 50 + "\n")


# Parse through the arguments
def arg_parser():
    # Creates the parser
    parser = argparse.ArgumentParser(description="MNIST digit recognition using CNN")

    # Add arguments
    parser.add_argument(
        "--download", type=bool, default=True, help="Download the MNIST dataset"
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
    save_examples(test_dataloader)

    return


if __name__ == "__main__":
    main()
