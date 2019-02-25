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
