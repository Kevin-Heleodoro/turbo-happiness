# Kevin Heleodoro - MNIST digit recognition using CNN (NextJournal - https://nextjournal.com/gkoehler/pytorch-mnist)

#  ------ Import Statements ------  #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

# ------ Global Variables ------  #
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)
mnist_mean = 0.1307
mnist_standard_deviation = 0.3081


# ------ Class Definitions ------  #
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ------ Function Definitions ------  #
def print_examples(test_loader):
    examples = enumerate(test_loader)
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

    print(f"Saving figure")
    fig.savefig("mnist_images.png")


def download_data():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/mnist2",
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
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "data/mnist2",
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
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )

    return train_loader, test_loader


# Saves the training data into text files to be used later
def save_training_data(train_losses, train_counter, test_losses, test_counter):
    with open("results/mnist2_train_losses.txt", "w") as f:
        for item in train_losses:
            f.write(f"{item}\n")

    with open("results/mnist2_train_counter.txt", "w") as f:
        for item in train_counter:
            f.write(f"{item}\n")

    with open("results/mnist2_test_losses.txt", "w") as f:
        for item in test_losses:
            f.write(f"{item}\n")

    with open("results/mnist2_test_counter.txt", "w") as f:
        for item in test_counter:
            f.write(f"{item}\n")


def load_training_data():
    with open("results/mnist2_train_losses.txt", "r") as f:
        train_losses = [float(line.strip()) for line in f]

    with open("results/mnist2_train_counter.txt", "r") as f:
        train_counter = [int(line.strip()) for line in f]

    with open("results/mnist2_test_losses.txt", "r") as f:
        test_losses = [float(line.strip()) for line in f]

    with open("results/mnist2_test_counter.txt", "r") as f:
        test_counter = [int(line.strip()) for line in f]

    return train_losses, train_counter, test_losses, test_counter


def plot_training_curve(train_losses, train_counter, test_losses, test_counter, epochs):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color="blue")
    plt.scatter(test_counter, test_losses, color="red")
    plt.legend(["Train Loss", "Test Loss"], loc="upper right")
    plt.xlabel("# of training examples seen")
    plt.ylabel("negative log likelihood loss")
    fig.savefig(f"results/mnist2_training_curve_{epochs}epochs.jpg")


def plot_prediction(example_data, output, epochs):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title(f"Prediction: {output.data.max(1, keepdim=True)[1][i].item()}")
        plt.xticks([])
        plt.yticks([])
    fig.savefig(f"results/mnist2_predictions_{epochs}epochs.jpg")


#  ------ Main Code ------  #
def main():
    train_loader, test_loader = download_data()

    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            f"\nTest set: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
        )

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % log_interval == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f})] \tLoss: {loss.item():.6f}"
                )
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                )
                torch.save(network.state_dict(), "results/mnist2_model.pth")
                torch.save(optimizer.state_dict(), "results/mnist2_optimizer.pth")

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    save_training_data(train_losses, train_counter, test_losses, test_counter)

    # Plotting the training curve
    plot_training_curve(
        train_losses, train_counter, test_losses, test_counter, n_epochs
    )

    _, (example_data, _) = next(enumerate(test_loader))

    with torch.no_grad():
        output = network(example_data)

    # Plotting the predictions
    plot_prediction(example_data, output, n_epochs)

    continued_network = MyNetwork()
    continued_optimizer = optim.SGD(
        network.parameters(), lr=learning_rate, momentum=momentum
    )

    # Load the model
    network_state_dict = torch.load("results/mnist2_model.pth")
    continued_network.load_state_dict(network_state_dict)

    # Load the optimizer
    optimzer_state_dict = torch.load("results/mnist2_optimizer.pth")
    continued_optimizer.load_state_dict(optimzer_state_dict)

    for i in range(4, 9):
        test_counter.append(i * len(train_loader.dataset))
        train(i)
        test()

    save_training_data(train_losses, train_counter, test_losses, test_counter)
    plot_training_curve(
        train_losses, train_counter, test_losses, test_counter, n_epochs + 5
    )


if __name__ == "__main__":
    main()
