# Initial design for neural network:
#   Inputs (I): 784 nodes (for each pixel value in dataset)
#   Hidden Layer 1 (H1): 16 nodes, each receiving input from all 784 input nodes
#   Hidden Layer 2 (H2): 16 nodes, each receiving input from all 16 H1 nodes
#   Outputs (0): 10 nodes (for each possible digit label)
import MNIST_Dataset as mnist
import numpy as np

class Neural_Network:
    def __init__(self, alpha):
        # Initialize each layer with random weights (values between 0-1)
        # Adding in a weight for the bias aswell
        self.weights_input_h1 = np.random.rand(16,784)
        self.weights_h1_h2 = np.random.rand(16,16)
        self.weights_h2_output = np.random.rand(10,16)

        # Learning rate is passed in and set
        self.lr = alpha

    # Sigmoid activation function (for feed forward algorithm)
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Derivative of sigmoid function (for backpropogation)
    def d_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def avgCost(self, guesses, targets):
        costs = []
        for i in range(len(guesses)):
            # print("(" + int(guesses[i]) + " - " + int(targets[i]) + ")^2")
            costs.append(((guesses[i] - targets[i]) ** 2) / 2)
        return sum(costs) / len(costs)

    def feed_forward(self, training_example):
        # Initial inputs fed into first layer of nodes and activated
        # (Input->Hidden_1)
        output_input_h1 = self.sigmoid(np.dot(self.weights_input_h1, training_example))

        # Inputs from Input->Hidden_1 fed into second layer of nodes and activated
        # (Hidden_1->Hidden_2)
        output_h1_h2 = self.sigmoid(np.dot(self.weights_h1_h2, output_input_h1))

        # Inputs from Hidden_1->Hidden_2 fed into third and final layer of nodes and
        # activated (Hidden_2->Output)
        output_h2_output = self.sigmoid(np.dot(self.weights_h2_output, output_h1_h2))

        # Returns the index of which output the network 'believes' the training
        # example represents. (indices: 0-9 for digits: 0-9)
        return np.argmax(output_h2_output)


    def backpropogation(self, training_examples, training_targets):
        # Initial inputs fed into first layer of nodes and activated
        # (Input->Hidden_1)
        for training_example in training_examples:
            output_input_h1 = self.sigmoid(np.dot(self.weights_input_h1, training_example))

            # Inputs from Input->Hidden_1 fed into second layer of nodes and activated
            # (Hidden_1->Hidden_2)
            output_h1_h2 = self.sigmoid(np.dot(self.weights_h1_h2, output_input_h1))

            # Inputs from Hidden_1->Hidden_2 fed into third and final layer of nodes and
            # activated (Hidden_2->Output)
            output_h2_output = self.sigmoid(np.dot(self.weights_h2_output, output_h1_h2))

            # print("Output between inputs and hidden 1:       \n", output_input_h1)
            # print("Output between hidden 1 and hidden 2:     \n", output_h1_h2)
            # print("Output between hidden 2 and final output: \n", output_h2_output, "\n")
        return None


# Loads the full 42,000 rows of test data
rawData = mnist.load_mnist();
# Gets the labels for each training example as a 784x1 vector
targets = mnist.get_targets(rawData);
# Gets the rest of the data as a 42000x784 matrix
data = mnist.get_data(rawData);


# Smaller training set
trainingData = data[:10]
trainingTargets = targets[:10]


nn = Neural_Network(0.1)
guesses = []
# For each entry (row) in the training set
for i in range(trainingData.shape[0]):
    # Add the guessed value
    guesses.append(nn.feed_forward(trainingData[i]))
    print("Actual Value:", trainingTargets[i], "    Guess value: ", nn.feed_forward(trainingData[i]))

print("Average cost of the final output: ", nn.avgCost(guesses,trainingTargets))
nn.backpropogation(trainingData, trainingTargets)
