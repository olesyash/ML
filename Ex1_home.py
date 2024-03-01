import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x):
    # The derivative of the sigmoid function here
    return (1 - sigmoid(x)) * sigmoid(x)


def forward(X, Wh, Wz):
    # Forward propagation step
    L1 = X @ Wh
    H = sigmoid(L1)
    L2 = H @ Wz
    Z = sigmoid(L2)
    return Z


def train(X, Y):
    # Define the number of epochs
    epochs = 5000
    # Define the size of the input layer, hidden layer, and output layer
    input_layer_size = 3
    hidden_layer_size = 7
    output_layer_size = 1
    learning_rate = 0.5
    Wh = np.random.uniform(size=(input_layer_size, hidden_layer_size))
    Wz = np.random.uniform(size=(hidden_layer_size, output_layer_size))

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
            return i, Wh, Wz

        # Backpropagation step
        # The error is the expected result minus actual result
        E = Y - Z
        print(f"Result Z: {Z}")
        print(f"Error shape: {E.shape}")
        print(f"Error: {E}")
        # Dz - that is multiply of the error and derivative of L2
        dZ = E * sigmoid_(L2)
        print("Dz shape: ")
        print(dZ.shape)
        # Dh is dz multiplied by derivative of L1
        dH = dZ @ Wz.T * sigmoid_(L1)
        print("Dh shape: ")
        print(dH.shape)
        # Update the weights
        # We need to transpose H to align with Dz shape, then multiply the H with dZ
        Wz += learning_rate * H.T @ dZ
        print(f"Wz shape: {Wz.shape}")
        # We need to transpose X to align with Dh shape, then multiply the X with dH
        Wh += learning_rate * X.T @ dH
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


# Example
X = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
Y = np.array([[0], [1], [1], [0]])

training_cycles, Wh, Wz = train(X, Y)
print(f"Training cycles: {training_cycles}")

new_input = np.array([[0.1, 0.1, 1.0], [0.0, 0.9, 1.0], [1.1, 0.0, 1.0], [1.1, 1.0, 1.0]])
new_output = forward(new_input,  Wh, Wz)
X = np.concatenate((X, new_input), axis=0)
new_output = np.array(classify(new_output))
Y = np.concatenate((Y, new_output), axis=0)

# Separate the inputs into two classes based on the labels
class_0 = X[Y[:, 0] == 0]
class_1 = X[Y[:, 0] == 1]

# Plot the two classes
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')

h = 0.01
x_range = np.arange(-0.1, 1.1, h)
y_range = np.arange(-0.1, 1.1, h)

# Add labels and a legend
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

# Show the plot
plt.show()
