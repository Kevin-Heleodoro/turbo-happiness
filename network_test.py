# Kevin Heleodoro - Testing the network using the MNIST dataset

# ----- Import Statements --------  #

import argparse
import torch
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transform2
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, InterpolationMode

from main import MyNetwork, print_border, load_data


# ----- Global Variables ---------  #

# ----- Class Definitions --------  #

# ----- Function Definitions -----  #


# Argument parser for the network test
def arg_parser():
    # Creates the parser
    parser = argparse.ArgumentParser(description="Run custom tests on the network")

    # Path to the saved model
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/main/mnist_model.pth",
        help="Path to the saved model",
    )

    # Use the MNIST dataset for testing
    parser.add_argument(
        "--mnist", type=bool, default=False, help="Use the MNIST dataset"
    )

    # Use custom 28x28 images for testing
    parser.add_argument(
        "--custom", type=bool, default=False, help="Use custom 28x28 images"
    )

    # Directory containing the custom images
    parser.add_argument(
        "--directory",
        type=str,
        default="data/custom",
        help="Directory containing the custom images",
    )

    return parser.parse_args()


# Run the tests on the network using custom images
def run_custom_samples(network, directory):
    print_border()
    print(f"Loading custom images from {directory}")

    mnist_mean = 0.1
    mnist_standard_deviation = 0.5

    # Load the custom images
    custom_data = ImageFolder(
        root=directory,
        transform=transform2.Compose(
            [
                transform2.Resize(
                    (28, 28), interpolation=InterpolationMode.BICUBIC, antialias=True
                ),
                transform2.Grayscale(),
                transform2.ToTensor(),
                transform2.Lambda(lambda x: 1 - x),
                transform2.Lambda(
                    lambda x: transform2.functional.adjust_sharpness(x, 2.0)
                ),
                transform2.Normalize((mnist_mean,), (mnist_standard_deviation,)),
            ]
        ),
    )
    print(f"Custom data classes: {custom_data.classes}")
    print(f"Custom data class to index: {custom_data.class_to_idx}")
    print(f"Custom data: {custom_data}")
    custom_loader = DataLoader(custom_data, batch_size=1)
    print(f"Number of custom images: {len(custom_data)}")

    # Test the network using the custom images
    network.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(custom_loader):
            output = network(data)
            print(f"Image #{i + 1}")
            network_values = []
            for j in range(10):
                network_values.append(round(output.data[0][j].item(), 2))
            print(f"Network Values: {network_values}")
            print(f"Prediction: {output.data.max(1, keepdim=True)[1][0].item()}")
            print(f"Ground Truth: {target[0]}\n")
            plt.imshow(data[0][0], cmap="gray", interpolation="none")
            plt.title(f"Prediction: {output.data.max(1, keepdim=True)[1][0].item()}")
            plt.xticks([])
            plt.yticks([])
            plt.show()


# Run the tests on the network using MNIST examples
def run_mnist_samples(network):
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

    # Parse through the arguments
    args = arg_parser()

    # Load the network from saved state
    print_border()
    print("Loading the network from saved state...")
    network = MyNetwork()
    model_path = args.model_path
    network.load_state_dict(torch.load(model_path))

    # Run the tests
    print_border()
    if args.mnist:
        print("Running tests using MNIST dataset ...")
        run_mnist_samples(network)

    if args.custom:
        print("Running tests using custom images ...")
        run_custom_samples(network, args.directory)

    print("Tests completed successfully")
    return


if __name__ == "__main__":
    main()
