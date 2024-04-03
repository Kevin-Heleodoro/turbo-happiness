# Kevin Heleodoro - CNN for MNIST Fashion dataset

# ----- Import Statements --------  #

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime

# ----- Global Variables ---------  #

LEARNING_RATE = 0.01
MOMENTUM = 0.5
EPOCHS = 5
MNIST_MEAN = 0.5
MNIST_STD_DEV = 0.5
BATCH_SIZE = 4
OPTIMIZER = "SGD"

# ----- Class Definitions --------  #


# Network for MNIST Fashion recognition
class FashionNetwork(nn.Module):
    def __init__(self):
        super(FashionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FashionNetwork2(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(FashionNetwork2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ----- Function Definitions -----  #


# Prints a border to separate sections
def print_border():
    print("\n" + "-" * 50 + "\n")


# Argument parser
def arg_parser():
    parser = argparse.ArgumentParser(description="Fashion MNIST recognition network")

    parser.add_argument(
        "--save-example",
        type=bool,
        default=False,
        help="Save first 4 examples from the dataset",
    )

    parser.add_argument(
        "--loadnetwork",
        type=str,
        default="",
        help="Path to MNIST Fashion trained model",
    )

    parser.add_argument(
        "--retrain",
        type=bool,
        default=False,
        help="Continue training the network passed in with the --load_network flag",
    )

    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train the network on"
    )

    parser.add_argument(
        "--batchsize", type=int, default=4, help="Batch size for loading dataset"
    )

    parser.add_argument(
        "--learningrate",
        type=float,
        default=0.01,
        help="Define the learning rate for the network",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        help="Declare optimizer for network (SGD, Adam, RMS)",
    )

    parser.add_argument(
        "--network",
        type=int,
        default=0,
        help="1: FashionNetwork, 2: FashionNetwork2",
    )

    parser.add_argument(
        "--auto-train",
        type=bool,
        default=False,
        help="Automatically train the network with different parameters",
    )

    return parser.parse_args()


# Load the MNIST Fashion dataset
def download_dataset(batch_size=BATCH_SIZE):
    root_dir = "data/mnist_fashion"

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD_DEV))]
    )

    training_data = torchvision.datasets.FashionMNIST(
        root=root_dir, train=True, download=True, transform=transform
    )
    testing_data = torchvision.datasets.FashionMNIST(
        root=root_dir, train=False, download=True, transform=transform
    )

    training_loader = torch.utils.data.DataLoader(
        training_data, batch_size=BATCH_SIZE, shuffle=True
    )
    testing_loader = torch.utils.data.DataLoader(
        testing_data, batch_size=BATCH_SIZE, shuffle=False
    )

    classes = (
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
    )

    print(f"Training set has {len(training_data)} images")
    print(f"Testing set has {len(testing_data)} images")

    return training_loader, testing_loader, classes


# Helper function for displaying an image grid
def matplot_make_grid(dataloader, one_channel=False):
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    img_grid = torchvision.utils.make_grid(images)

    fig = plt.figure()
    if one_channel:
        img_grid = img_grid.mean(dim=0)

    img_grid = img_grid / 2 + 0.5
    npimg = img_grid.numpy()

    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.title("Fashion MNIST Images")
    plt.show()

    file_path = f"results/main/task_4/images/sample_images.png"
    fig.savefig(file_path)

    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_dataloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = network(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if i % 1000 == 999:
            last_loss = running_loss / 1000
            print(f"Epoch: {epoch_idx + 1}, Batch: {i + 1}, Loss: {last_loss:.3f}")
            tb_x = epoch_idx * len(training_dataloader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


# Train network
def train_network(network, optimizer, train_loader, epoch, train_losses, train_counter):
    print_border()
    print("Training the network...")

    network.train()
    log_interval = 1000
    loss_fn = nn.CrossEntropyLoss()

    for idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = loss_fn(output, target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if idx % log_interval == 0:
            print(
                f"Training Epoch {epoch} [{idx * len(data)}/{len(train_loader.dataset)} ({100. * idx / len(train_loader):.0f}%)] \tLoss: {loss.item():.6f} "
            )

            train_losses.append(loss.item())
            train_counter.append(
                (idx * BATCH_SIZE) + ((epoch - 1) * len(train_loader.dataset))
            )

            model_path = f"results/main/task_4/fashion_model_{EPOCHS}_epochs.pth"
            optimizer_path = (
                f"results/main/task_4/fashion_{OPTIMIZER}_optim_{EPOCHS}_epochs.pth"
            )

            torch.save(network.state_dict(), model_path)
            torch.save(optimizer.state_dict(), optimizer_path)

    print("Training complete!")


# Test the network
def test_network(
    test_loader, network, test_losses, epoch, batch_size=BATCH_SIZE, total_epochs=EPOCHS
):
    print_border()
    print("Testing the network...")

    network.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    print(f"test loss: {test_loss}")
    test_loss /= len(test_loader.dataset)
    print(f"test loss averaged: {test_loss}")
    test_losses.append(test_loss)

    print(
        f"\nTest set (Epoch {epoch}) - Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n"
    )

    # Save the results into a text file:
    with open("results/main/task_4/results.txt", "a") as f:
        f.write(
            f"Epoch: {epoch}, Test loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%), Batch Size: {batch_size}, Total Epochs: {total_epochs}, Dropout Rate: {network.dropout}\n"
        )


# Logic for testing the training and testing loops with different values
def run_main_loop(total_epochs, dropout_rate, batch_size, args, network_choice=1):
    print_border()
    print("Starting training loop")

    BATCH_SIZE = batch_size
    EPOCHS = total_epochs
    LEARNING_RATE = args.learningrate  # 0.01 default
    OPTIMIZER = args.optimizer  # SGD default

    print(
        f"Running with {total_epochs} epochs, {dropout_rate} dropout rate, and {batch_size} batch size"
    )

    # Load data
    print_border()
    print("Loading the MNIST Fashion dataset...")
    train_loader, test_loader, classes = download_dataset(batch_size)

    # Display data
    # print_border()
    # print("Displaying the MNIST Fashion dataset...")
    # matplot_make_grid(train_loader)

    # Load the network
    print_border()
    print("Loading the network...")
    if network_choice == 1:
        network = FashionNetwork()
    elif network_choice == 2:
        network = FashionNetwork2(dropout_rate=dropout_rate)
    if len(args.loadnetwork) > 1:
        try:
            model_path = args.loadnetwork
            network.load_state_dict(model_path)
        except:
            print(f"Invalid model path: {args.loadnetwork}")
            return -1
    print(network)

    match OPTIMIZER:
        case "SGD":
            optimizer = torch.optim.SGD(
                network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
            )
        case "Adam":
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        case "RMS":
            optimizer = torch.optim.RMSprop(
                network.parameters(), lr=1e-3, momentum=MOMENTUM
            )
        case _:
            optimizer = torch.optim.SGD(
                network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
            )

    print(f"Optimizer set to {optimizer}")

    # Train the network
    print_border()

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(EPOCHS + 1)]

    print("Running a pre-training test...")
    test_network(
        test_loader,
        network,
        test_losses,
        0,
        batch_size=batch_size,
        total_epochs=total_epochs,
    )

    starting_epoch = 1

    for epoch in range(starting_epoch, EPOCHS + 1):
        train_network(
            network, optimizer, train_loader, epoch, train_losses, train_counter
        )
        test_network(
            test_loader,
            network,
            test_losses,
            epoch,
            batch_size=batch_size,
            total_epochs=total_epochs,
        )


# ----- Main Code ----------------  #


def main():
    print_border()
    print("Starting the MNIST Fashion recognition program...")

    args = arg_parser()

    if args.auto_train:
        batch_size_list = [4, 8, 16, 32, 64]
        dropout_rate_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        num_of_epochs_list = [5, 10, 15, 20, 25]

        for bs in batch_size_list:
            run_main_loop(3, 0.5, bs, args, 2)
        for dr in dropout_rate_list:
            run_main_loop(3, dr, 4, args, 2)
        for ep in num_of_epochs_list:
            run_main_loop(ep, 0.5, 4, args, 2)
    else:
        BATCH_SIZE = args.batchsize
        EPOCHS = args.epochs
        LEARNING_RATE = args.learningrate
        OPTIMIZER = args.optimizer

        run_main_loop(EPOCHS, 0.5, BATCH_SIZE, args, args.network)

    return


if __name__ == "__main__":
    main()
