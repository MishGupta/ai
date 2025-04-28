import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # Step 1: Read number of inputs
    N = int(input("Enter number of inputs (N): "))
    
    # Step 2: Generate random binary input vector
    X = np.random.randint(0, 2, size=(N, 1))
    print("Input X:", X.T)  # Transposed for better viewing

    # Step 3: Initialize random weights and biases
    W1 = np.random.rand(3, N)    # Weights from input to first hidden layer
    b1 = np.random.rand(3, 1)    # Biases for first hidden layer

    W2 = np.random.rand(2, 3)    # Weights from first hidden to second hidden layer
    b2 = np.random.rand(2, 1)    # Biases for second hidden layer

    W3 = np.random.rand(1, 2)    # Weights from second hidden to output layer
    b3 = np.random.rand(1, 1)    # Bias for output layer

    # Step 4: Forward pass (only one step)
    Z1 = sigmoid(np.dot(W1, X) + b1)   # First hidden layer output
    Z2 = sigmoid(np.dot(W2, Z1) + b2)  # Second hidden layer output
    output = sigmoid(np.dot(W3, Z2) + b3)  # Final output

    # Step 5: Display final outputs
    print("\nFinal Output:", output)
    print("\nFinal Weight Matrices and Biases:")
    print("W1 (Input -> Hidden 1):\n", W1)
    print("b1 (Bias Hidden 1):\n", b1)
    print("W2 (Hidden 1 -> Hidden 2):\n", W2)
    print("b2 (Bias Hidden 2):\n", b2)
    print("W3 (Hidden 2 -> Output):\n", W3)
    print("b3 (Bias Output):\n", b3)
    
    # Step 6: Number of steps
    print("\nNumber of steps: 1 (One forward pass)")

if __name__ == "__main__":
    main()
