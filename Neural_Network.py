import MNIST_Dataset as mnist

# Loads the full 42,000 rows of test data
rawData = mnist.load_mnist();
# Gets the labels for each training example as a 784x1 vector
targets = mnist.get_targets(rawData);
# Gets the rest of the data as a 42000x784 matrix
data = mnist.get_data(rawData);
