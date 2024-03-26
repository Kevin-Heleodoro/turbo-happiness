# Recognition using Deep Networks

Kevin Heleodoro
March 21, 2024

## Task 1

[Tutorial on setting up dnn using PyTorch and MNIST dataset](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)

[MNIST Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST)

[Loading data in PyTorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

[Building Neural Networks in PyTorch](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

[Training model in PyTorch](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

[Saving and loading model in PyTorch](https://pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

[MNIST Tutorial](https://nextjournal.com/gkoehler/pytorch-mnist)

### PyTorch Tutorial

Followed the [PyTorch tutorial](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) to build `mnist1.py`

Results from using the PyTorch tutorial on the MNIST dataset:

`Accuracy: 73.4%, Avg loss: 1.585726 ` <- 5 epochs

### NextJournal Tutorial

Followed the [NextJournal tutorial](https://nextjournal.com/gkoehler/pytorch-mnist) to build `mnist2.py`

Results from using the NextJournal MNIST tutorial:

`Test set: Avg loss: 2.3316, Accuracy: 1137/10000 (11%)` <- initial no training

`Test set: Avg loss: 0.1011, Accuracy: 9668/10000 (97%)` <- 3 epochs

`Test set: Avg loss: 0.0595, Accuracy: 9822/10000 (98%)` <- 8 epochs

### Main.py

Used a combination of both the tutorials to download the data and save the test examples.

Used [PyTorch documentation on building graphs](https://pytorch.org/docs/stable/nn.html) to set up network.

Used [Tensorboard](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html) to create network diagram.

First test run (pre-training):

> `Test set: Avg loss: 2.3078, Accuracy: 1078/10000 (11%)`

Epoch 1:

> `Test set: Avg loss: 0.1555, Accuracy: 9623/10000 (96%)`

Epoch 2:

> `Test set: Avg loss: 0.0969, Accuracy: 9759/10000 (98%)`

Epoch 3:

> `Test set: Avg loss: 0.0751, Accuracy: 9807/10000 (98%)`

Epoch 4:

> `Test set: Avg loss: 0.0649, Accuracy: 9842/10000 (98%)`

Epoch 5:

> `Test set: Avg loss: 0.0566, Accuracy: 9866/10000 (99%)`

### network_test.py

Loading the `state_dict` that was saved from main.py to test on the first 10 example images from the MNIST testing dataset.

Results from the command line:
![](/results/main/mnist_prediction_cli.png)

Results in a figure
![](/results/main/predicted_images.png)
