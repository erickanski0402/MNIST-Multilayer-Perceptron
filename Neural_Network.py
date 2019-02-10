# Initial design for neural network:
#   Inputs (I): 784 nodes (for each pixel value in dataset)
#   Hidden Layer 1 (H1): 16 nodes, each receiving input from all 784 input nodes
#   Hidden Layer 2 (H2): 16 nodes, each receiving input from all 16 H1 nodes
#   Outputs (0): 10 nodes (for each possible digit label)
#

import MNIST_Dataset as mnist

# Loads the full 42,000 rows of test data
rawData = mnist.load_mnist();
# Gets the labels for each training example as a 784x1 vector
targets = mnist.get_targets(rawData);
# Gets the rest of the data as a 42000x784 matrix
data = mnist.get_data(rawData);
