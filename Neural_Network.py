# Initial design for neural network:
#   Inputs (I): 784 nodes (for each pixel value in dataset)
#   Hidden Layer 1 (H1): 16 nodes, each receiving input from all 784 input nodes
#   Hidden Layer 2 (H2): 16 nodes, each receiving input from all 16 H1 nodes
#   Outputs (0): 10 nodes (for each possible digit label)

import MNIST_Dataset as mnist
import numpy as np


# Initialize each layer with random weights (values between 0-1)
weights_input_h1 = np.random.rand(16,784)
weights_h1_h2 = np.random.rand(16,16)
weights_h2_output = np.random.rand(10,16)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def feed_forward(training_example):
    # Initial inputs fed into first layer of nodes and activated
    # (Input->Hidden_1)
    output_input_h1 = sigmoid(np.dot(weights_input_h1, training_example))

    # Inputs from Input->Hidden_1 fed into second layer of nodes and activated
    # (Hidden_1->Hidden_2)
    output_h1_h2 = sigmoid(np.dot(weights_h1_h2, output_input_h1))

    # Inputs from Hidden_1->Hidden_2 fed into third and final layer of nodes and
    # activated (Hidden_2->Output)
    output_h2_output = sigmoid(np.dot(weights_h2_output, output_h1_h2))

    # Returns the index of which output the network 'believes' the training
    # example represents. (indices: 0-9 for digits: 0-9)
    return np.argmax(output_h2_output)


# Loads the full 42,000 rows of test data
rawData = mnist.load_mnist();
# Gets the labels for each training example as a 784x1 vector
targets = mnist.get_targets(rawData);
# Gets the rest of the data as a 42000x784 matrix
data = mnist.get_data(rawData);


# Smaller training set
trainingData = data[:10]
trainingTargets = targets[:10]


# # Feed forward
# # Initial inputs fed into first layer of nodes and activated (Input->Hidden_1)
# output_input_h1 = sigmoid(np.dot(weights_input_h1, trainingData))
#
# # Inputs from Input->Hidden_1 fed into second layer of nodes and activated
# # (Hidden_1->Hidden_2)
# output_h1_h2 = sigmoid(np.dot(weights_h1_h2, output_input_h1))
#
# # Inputs from Hidden_1->Hidden_2 fed into third and final layer of nodes and
# # activated (Hidden_2->Output)
# output_h2_output = sigmoid(np.dot(weights_h2_output, output_h1_h2))
# print("Actual Digit: ", trainingTargets)
# print("Guessed Digit: ", np.argmax(output_h2_output))


for i in range(trainingData.shape[0]):
    print("Actual Value:", trainingTargets[i], "    Guess value: ", feed_forward(trainingData[i]))
