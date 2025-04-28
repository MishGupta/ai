import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    print("Simple MLP with 4 Inputs, 1 Hidden Layer, 2 Outputs")

    # Fixed number of inputs
    N = 4
    X = np.random.randint(0, 2, size=(N, 1))
    print("\nInput X:", X.T)

    # Random initialization
    W1 = np.random.rand(3, N)    # Hidden layer: 3 neurons
    b1 = np.random.rand(3, 1)

    W2 = np.random.rand(2, 3)    # Output layer: 2 neurons
    b2 = np.random.rand(2, 1)

    # One forward pass
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    output = sigmoid(Z2)

    print("\nFinal Output:", output)
    print("Binary Output (Thresholded):", (output > 0.5).astype(int))

    print("\nFinal Weight Matrices and Bias Values:")
    print("W1 (Input to Hidden Layer):\n", W1)
    print("b1 (Hidden Layer Biases):\n", b1)
    print("W2 (Hidden to Output Layer):\n", W2)
    print("b2 (Output Layer Biases):\n", b2)

    print("\nNumber of Steps:", 1)

if __name__ == "__main__":
    main()
