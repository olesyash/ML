import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x):
    # The derivative of the sigmoid function here
    return (1 - sigmoid(x)) * sigmoid(x)


def train():
    # Define the number of epochs
    epochs = 1000
    # Define the size of the input layer, hidden layer, and output layer
    inputLayerSize = 3
    hiddenLayerSize = 4
    outputLayerSize = 1
    Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
    Wz = np.random.uniform(size=(hiddenLayerSize, outputLayerSize))

    for i in range(epochs):
        # Forward propagation step
        L1 = X @ Wh
        H = sigmoid(L1)
        L2 = H @ Wz
        Z = sigmoid(L2)
        print(f"Z shape: {Z.shape}")

        # Check the response in manner of bigger or smaller than 0.5
        new_res = classify(Z)
        if np.array_equal(new_res, Y):
            # When we reach desired answer, break and return the training cycles
            return i

        # Backpropagation step
        E = Y - Z
        print(f"Result Z: {Z}")
        print(f"Error shape: {E.shape}")
        print(f"Error: {E}")
        dZ = E * sigmoid_(L2)
        print("Dz shape: ")
        print(dZ.shape)
        dH = dZ @ Wz.T * sigmoid_(L1)
        print("Dh shape: ")
        print(dH.shape)
        # Update the weights
        Wz += H.T @ dZ
        print(f"Wz shape: {Wz.shape}")
        Wh += X.T @ dH
        print(f"Wh shape: {Wh.shape}")


def classify(res):
    """
    If the result is bigger than 0.5 consider as 1, if lower consider as 0
    :param res: original result vector
    :return: new result vector
    """
    new_res = []
    for i in range(0, len(res)):
        if res[i] > 0.5:
            new_res.append([1])
        else:
            new_res.append([0])
    print(f"Result: {new_res}")
    return np.array(new_res)


X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
Y = np.array([[0], [1], [1], [0]])

# Separate the inputs into two classes based on the labels
class_0 = X[Y[:, 0] == 0]
class_1 = X[Y[:, 0] == 1]

# Plot the two classes
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')

# Add labels and a legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

# Show the plot
# plt.show()

training_cycles = train()
print(f"Training cycles: {training_cycles}")
