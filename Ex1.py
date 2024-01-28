import numpy as np
import matplotlib.pyplot as plt


# Define the update_weights function
def update_weights(m, b, X, Y, learning_rate):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        # Calculate partial derivatives
        m_deriv += -2*X[i] * (Y[i]-(m*X[i] + b))
        b_deriv += -2*(Y[i] - (m*X[i] + b))

    # Update m and b
    m -= (m_deriv / float(N)) * learning_rate
    b -= (b_deriv / float(N)) * learning_rate

    return m, b

# Initialize parameters
m = 0
b = 0
learning_rate = 0.01

# Sample dataset
X = np.array([1, 2, 3, 4, 5])
Y = np.array([5, 7, 9, 11, 13])

# Perform gradient descent
iterations = 1000
for i in range(iterations):
    m, b = update_weights(m, b, X, Y, learning_rate)

# Print final parameters
final_m, final_b = m, b
print("Final slope (m):", final_m)
print("Final y-intercept (b):", final_b)

# Plotting the results
plt.scatter(X, Y, color='red', label='Data Points')
plt.plot(X, final_m * X + final_b, color='blue', label='Fitted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()
