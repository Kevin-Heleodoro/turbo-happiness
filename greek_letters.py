# Kevin Heleodoro - Use the MNIST trained network to recognize Greek letters

# ----- Import Statements --------  #

import torch
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt

from main import MyNetwork, print_border


# ----- Global Variables ---------  #

learning_rate = 0.1
momentum = 0.8
n_epochs = 5

# ----- Class Definitions --------  #


# Greek dataset transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36 / 128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


# ----- Function Definitions -----  #


# Load the Greek letters dataset
def load_greek_data(directory):
    print(f"Loading Greek letters from {directory}...")

    mnist_mean = 0.1307
    mnist_standard_deviation = 0.3081

    greek_data = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            directory,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    GreekTransform(),
                    torchvision.transforms.Normalize(
                        (mnist_mean,), (mnist_standard_deviation,)
                    ),
                ]
            ),
        ),
        batch_size=5,
        shuffle=True,
    )

    return greek_data


# Train the network
def train_network(
    network, optimizer, train_dataloader, train_losses, train_counter, epoch
):
    network.train()

    log_interval = 1

    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_dataloader.dataset)} "
                f"({100. * batch_idx / len(train_dataloader):.0f}%)]\tLoss: {loss.item():.6f}"
            )
            train_losses.append(loss.item())
            count = (epoch - 1) * len(train_dataloader.dataset) + batch_idx * len(data)
            train_counter.append(
                count
                # (batch_idx * 64) + ((epoch - 1) * len(train_dataloader.dataset))
            )
    print("Training complete")


def test_network(network, test_dataloader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_dataloader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_dataloader.dataset)} "
        f"({100. * correct / len(test_dataloader.dataset):.2f}%)\n"
    )


# Plot the training curve
def plt_training_curve(
    train_losses, train_counter, test_losses, test_counter, n_epochs
):

    print_border()
    print("Plotting the training curve...")

    print("Train Counter: ", len(train_counter))
    print("Train Losses: ", len(train_losses))
    print("Test Counter: ", len(test_counter))
    print("Test Losses: ", len(test_losses))

    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="pink")
    plt.scatter(test_counter, test_losses, color="green")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.title("MNIST CNN Training Curve")
    plt.tight_layout()
    plt.savefig("results/main/task_3/greek_training_curve.png")
    print("Training curve plotted successfully")


# ----- Main Code ----------------  #


# Use the trained network to recognize Greek letters
def main():
    print_border()
    print("Greek Letter Recognition")

    # Load the trained model
    print_border()
    print("Loading the trained model...")
    network = MyNetwork()
    model_path = "results/main/mnist_model.pth"
    network.load_state_dict(torch.load(model_path))

    # Freeze network weights
    print_border()
    print("Freezing the network weights...")
    for param in network.parameters():
        param.requires_grad = False

    # Replace the last layer
    print_border()
    print("Replacing the last layer with a new layer with three nodes...")
    in_features = network.fc2.in_features
    network.fc2 = torch.nn.Linear(in_features, 3)
    network.fc2.requires_grad = True

    # Load the Greek letters dataset
    print_border()
    print("Loading the Greek letters dataset...")
    greek_train_dir = "data/greek_train"
    greek_train = load_greek_data(greek_train_dir)
    greek_test_dir = "data/greek_test"
    greek_test = load_greek_data(greek_test_dir)

    # Train the network
    print_border()
    print("Training the network...")
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(greek_test.dataset) for i in range(n_epochs + 1)]
    optimizer = torch.optim.SGD(
        network.fc2.parameters(), lr=learning_rate, momentum=momentum
    )

    print("Running pre-training test...")
    test_network(network, greek_test, test_losses)

    for epoch in range(1, n_epochs + 1):
        train_network(
            network, optimizer, greek_train, train_losses, train_counter, epoch
        )
        test_network(network, greek_test, test_losses)

    # Plot the training curve
    plt_training_curve(train_losses, train_counter, test_losses, test_counter, n_epochs)


if __name__ == "__main__":
    main()
