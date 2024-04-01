# Kevin Heleodoro - Examine the network's architecture

# ----- Import Statements --------  #

import cv2
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from main import MyNetwork, print_border, load_data

# ----- Global Variables ---------  #

# ----- Class Definitions --------  #

# ----- Function Definitions -----  #


# Analyze the given layer of the network
def analyze_layer(layer):
    print(f"Layer: {layer}")
    print(f"Layer weight shape: {layer.weight.shape}")

    # Visualize the filters using pyplot
    plt.figure()
    print("Visualizing the filters...")
    for i in range(10):
        plt.subplot(3, 4, i + 1)
        plt.imshow(layer.weight[i][0].detach().numpy())
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
    plt.show()


# Show effects of the first layer using OpenCV filter 2D function
def show_effects(layer, image):
    image_tensor = image.squeeze().numpy()

    print("Iterating through filters in layer...")
    plt.figure()
    with torch.no_grad():
        for i in range(10):
            filter_kernel = layer.weight[i][0].cpu().detach().numpy()
            filtered_image = cv2.filter2D(image_tensor, -1, filter_kernel)
            plt.subplot(5, 4, 2 * i + 1)
            plt.imshow(filter_kernel, cmap="gray")
            plt.axis("off")
            plt.xticks([])
            plt.yticks([])

            plt.subplot(5, 4, 2 * i + 2)
            plt.imshow(filtered_image, cmap="gray")
            plt.axis("off")
            plt.xticks([])
            plt.yticks([])

    print("Filters applied successfully!")
    plt.tight_layout()
    plt.show()


# ----- Main Code ----------------  #


# Examine the network's architecture
def main():
    print_border()
    print("Examine the network's architecture")

    # Load training data
    print_border()
    print("Loading training data...")
    train_loader, _ = load_data()

    # Load the network
    print_border()
    print("Loading the network...")
    network = MyNetwork()
    model_path = "results/main/mnist_model.pth"
    network.load_state_dict(torch.load(model_path))

    # Print the network architecture
    print_border()
    print(f"Model loaded from: {model_path}")
    print(network)

    # Analyze first layer
    print_border()
    print("Analyzing the first layer...")
    analyze_layer(network.conv1)

    # Apply filters to first image
    print_border()
    print("Applying first layer filters to the test image...")
    image_to_filter = train_loader.dataset[0][0]
    show_effects(network.conv1, image_to_filter)

    print("Terminating the program")


if __name__ == "__main__":
    main()
