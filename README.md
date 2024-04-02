# Recognition using Deep Networks

Kevin Heleodoro
March 21, 2024

## Instructions:

To execute network training run the following command:

```sh
python main.py --train_work True
```

You can also pass the following options to the `main.py` file:

```sh
usage: main.py [-h] [--download DOWNLOAD] [--save_example SAVE_EXAMPLE] [--visualize VISUALIZE]
               [--train_network TRAIN_NETWORK]

MNIST digit recognition using CNN

options:
  -h, --help            show this help message and exit
  --download DOWNLOAD   Download the MNIST dataset
  --save_example SAVE_EXAMPLE
                        Save first 6 examples from the test data
  --visualize VISUALIZE
                        Visualize the network using TensorBoard
  --train_network TRAIN_NETWORK
                        Train the network using the MNIST dataset
```

To execute the network test on handwritten numbers run the following command:

```sh
python network_test.py --custom True --directory "data/custom"
```

To examine the network architecture run the following command:

```sh
python examine_network.py
```

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

### Testing on custom inputs

Downloaded the [Image Magick CLI tool](https://imagemagick.org/script/download.php) to resize the images I hand wrote.

```sh
magick '*.jpg[28x28]' resize_%03d.png
```

Original Image:
![](results/main/numbers_preformat.jpg)

Example of resized:
![](results/main/numbers/resized/resized_000.jpg)

Used [torchvision.datasets.ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html) for custom image loading.

The first round of custom input resulted in only 1/10 correct predictions.
![](results/main/numbers/round%201/custom_predictions_1.png)

Added `torchvision.transforms.Normalize((0.1307,), (0.3081,))` to the transform for the custom image set.
-> Still only 1/10 correct predictions.

The resulting images (results/main/numbers/round_1) are very blurry.

After inverting the intensities using `torchvision.transforms.lambda()`:

```sh
Loading custom images from data/original
Custom data classes: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
Custom data class to index: {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
Custom data: Dataset ImageFolder
    Number of datapoints: 10
    Root location: data/original
    StandardTransform
Transform: Compose(
               Resize(size=(28, 28), interpolation=bicubic, max_size=None, antialias=warn)
               Grayscale(num_output_channels=1)
               ToTensor()
               Lambda()
               Normalize(mean=(0.1,), std=(0.5,))
           )
Number of custom images: 10
Image #1
Network Values: [-1.32, -3.08, -2.57, -1.8, -3.27, -2.45, -3.03, -3.16, -2.26, -2.07]
Prediction: 0
Ground Truth: 0

Image #2
Network Values: [-2.26, -1.77, -2.09, -2.01, -3.06, -2.72, -2.9, -2.92, -1.63, -2.99]
Prediction: 8
Ground Truth: 1

Image #3
Network Values: [-2.68, -2.66, -1.5, -2.35, -2.29, -3.04, -2.85, -2.61, -1.58, -2.92]
Prediction: 2
Ground Truth: 2

Image #4
Network Values: [-3.18, -3.44, -3.15, -0.94, -3.43, -1.87, -3.13, -3.33, -1.91, -2.56]
Prediction: 3
Ground Truth: 3

Image #5
Network Values: [-3.57, -2.63, -3.03, -2.13, -1.94, -2.17, -3.45, -1.86, -1.78, -2.15]
Prediction: 8
Ground Truth: 4

Image #6
Network Values: [-3.09, -3.54, -2.65, -2.0, -3.42, -1.21, -2.71, -3.47, -1.41, -3.07]
Prediction: 5
Ground Truth: 5

Image #7
Network Values: [-2.41, -3.52, -2.97, -2.88, -2.75, -2.02, -1.33, -3.88, -1.44, -2.88]
Prediction: 6
Ground Truth: 6

Image #8
Network Values: [-2.52, -2.46, -2.42, -2.08, -2.84, -2.63, -3.55, -1.86, -1.65, -2.18]
Prediction: 8
Ground Truth: 7

Image #9
Network Values: [-2.6, -3.32, -2.69, -2.11, -3.57, -2.19, -2.77, -3.45, -0.95, -2.52]
Prediction: 8
Ground Truth: 8

Image #10
Network Values: [-2.67, -2.03, -2.51, -1.98, -2.46, -2.86, -3.68, -1.92, -1.88, -2.17]
Prediction: 8
Ground Truth: 9
```

Using [`torchvision.transforms.functional.adjust_sharpness()`](https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_sharpness.html) improved the results slightly. However the digits are still blurry when seeing the matplotlib output.

Switched to using `torchvision.transforms.v2` per [PyTorch's recommendation](https://pytorch.org/vision/stable/transforms.html).

## Task 2

Examining the network using `examine_network.py`.

```sh
Model loaded from:  results/main/mnist_model.pth
MyNetwork(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (dropout): Dropout(p=0.5, inplace=False)
  (fc1): Linear(in_features=320, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
)
```

Convolution layer 1 filters:
![](results/main/task_2/conv1_filters.png)

Filters applied to first image in training set:
![](results/main/task_2/conv1_filters_to_image_1.png)

## Task 3

Transfer learning on Greek Letters

First iteration using 5 epochs, 0.01 learning rate, 0.5 momentum:

```sh
Training the network...
Running pre-training test...

Test set: Average loss: 1.0932, Accuracy: 3/9 (33.33%)

Train Epoch: 1 [0/27 (0%)]	Loss: 1.047323
Train Epoch: 1 [5/27 (17%)]	Loss: 1.050199
Train Epoch: 1 [10/27 (33%)]	Loss: 0.820251
Train Epoch: 1 [15/27 (50%)]	Loss: 1.310539
Train Epoch: 1 [20/27 (67%)]	Loss: 0.773852
Train Epoch: 1 [10/27 (83%)]	Loss: 1.201905
Training complete

Test set: Average loss: 0.9173, Accuracy: 5/9 (55.56%)

Train Epoch: 2 [0/27 (0%)]	Loss: 0.728469
Train Epoch: 2 [5/27 (17%)]	Loss: 0.714651
Train Epoch: 2 [10/27 (33%)]	Loss: 0.916280
Train Epoch: 2 [15/27 (50%)]	Loss: 0.836977
Train Epoch: 2 [20/27 (67%)]	Loss: 0.761337
Train Epoch: 2 [10/27 (83%)]	Loss: 0.794498
Training complete

Test set: Average loss: 0.8028, Accuracy: 7/9 (77.78%)

Train Epoch: 3 [0/27 (0%)]	Loss: 0.642418
Train Epoch: 3 [5/27 (17%)]	Loss: 0.516627
Train Epoch: 3 [10/27 (33%)]	Loss: 0.601593
Train Epoch: 3 [15/27 (50%)]	Loss: 0.645809
Train Epoch: 3 [20/27 (67%)]	Loss: 0.531400
Train Epoch: 3 [10/27 (83%)]	Loss: 0.764568
Training complete

Test set: Average loss: 0.7297, Accuracy: 7/9 (77.78%)

Train Epoch: 4 [0/27 (0%)]	Loss: 0.474293
Train Epoch: 4 [5/27 (17%)]	Loss: 0.405608
Train Epoch: 4 [10/27 (33%)]	Loss: 0.442747
Train Epoch: 4 [15/27 (50%)]	Loss: 0.739589
Train Epoch: 4 [20/27 (67%)]	Loss: 0.396530
Train Epoch: 4 [10/27 (83%)]	Loss: 0.572971
Training complete

Test set: Average loss: 0.6900, Accuracy: 7/9 (77.78%)

Train Epoch: 5 [0/27 (0%)]	Loss: 0.482386
Train Epoch: 5 [5/27 (17%)]	Loss: 0.478900
Train Epoch: 5 [10/27 (33%)]	Loss: 0.384140
Train Epoch: 5 [15/27 (50%)]	Loss: 0.351912
Train Epoch: 5 [20/27 (67%)]	Loss: 0.307006
Train Epoch: 5 [10/27 (83%)]	Loss: 0.445835
Training complete

Test set: Average loss: 0.6486, Accuracy: 7/9 (77.78%)
```

Increasing the number of epochs to 10 and then 20 resulted in a lower accuracy score.

Reduced learning rate to 0.1 and momentum to 0.8 which cut the average loss down by ~50%.

```sh
Train Epoch: 5 [0/27 (0%)]	Loss: 0.008052
Train Epoch: 5 [5/27 (17%)]	Loss: 0.026820
Train Epoch: 5 [10/27 (33%)]	Loss: 0.010399
Train Epoch: 5 [15/27 (50%)]	Loss: 0.006687
Train Epoch: 5 [20/27 (67%)]	Loss: 0.014652
Train Epoch: 5 [10/27 (83%)]	Loss: 0.002258
Training complete

Test set: Average loss: 0.1444, Accuracy: 8/9 (88.89%)
```

**Results varied from 7-8/9 correct answer**

The Training curve does not follow any kind of pattern or have any consistency between runs.
![](results/main/task_3/greek_training_curve.png)

## Task 4

Using the [pytorch tutorial](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html?highlight=nn%20crossentropyloss) as a starting point for the MNIST Fasion dataset Network
