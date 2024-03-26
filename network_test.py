# Kevin Heleodoro - Testing the network using the MNIST dataset

# ----- Import Statements --------  #

import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from main import MyNetwork, print_border, load_data


# ----- Global Variables ---------  #

# ----- Class Definitions --------  #

# ----- Function Definitions -----  #


# Run the tests on the network
def run_tests(network):
    # Load the MNIST dataset
    _, test_loader = load_data()

    # Test the network using the first n examples from the MNIST dataset
    n = 10
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(f"Example data shape: {example_data.shape}")
    test_set = example_data[:n]

    print_border()
    print(
        f"Running tests on the network using the first {n} examples from the MNIST dataset"
    )

    # Test the network
    network.eval()
    with torch.no_grad():
        output = network(test_set)
    print(f"Output shape: {output.shape}\n")

    # Display the network's predictions
    fig = plt.figure()
    for i in range(9):
        print(f"Image #{i + 1}")
        network_values = []
        for j in range(10):
            network_values.append(round(output.data[i][j].item(), 2))
        print(f"Network Values: {network_values}")
        print(f"Prediction: {output.data.max(1, keepdim=True)[1][i].item()}")
        print(f"Ground Truth: {example_targets[i]}\n")
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(test_set[i][0], cmap="gray", interpolation="none")
        plt.title(f"Prediction: {output.data.max(1, keepdim=True)[1][i].item()}")
        plt.xticks([])
        plt.yticks([])

    print(f"Saving figure")
    fig.savefig("results/main/predicted_images.png")
    plt.show()

    print_border()

    return


# ----- Main Code ----------------  #


# Tests the network using the first n examples from the MNIST dataset
def main():
    print_border()
    print("Testing the network using the MNIST dataset")

    # Load the network from saved state
    print_border()
    print("Loading the network from saved state...")
    network = MyNetwork()
    network.load_state_dict(torch.load("results/main/mnist_model.pth"))

    # Run the tests
    print_border()
    print("Running tests...")
    run_tests(network)

    print("Tests completed successfully")
    return


if __name__ == "__main__":
    main()
