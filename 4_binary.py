import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def main():
    # Fixed number of inputs
    N = 4
    X = np.random.randint(0, 2, size=(N,))
    Y = np.random.randint(0, 2, size=(2,))  # Expected output - 2 binary outputs
    print("Input X:", X)
    print("Target Y:", Y)

    # Random initialization
    W1 = np.random.rand(3, N)    # Hidden layer: 3 neurons
    b1 = np.random.rand(3)

    W2 = np.random.rand(2, 3)    # Output layer: 2 neurons
    b2 = np.random.rand(2)

    # Training parameters
    learning_rate = 0.1
    max_steps = 1000
    
    # Training loop
    steps = 0
    for step in range(max_steps):
        # Forward pass
        Z1 = np.dot(W1, X) + b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        
        # Compute loss
        loss = np.mean(np.square(A2 - Y))
        
        # Early stopping
        if loss < 0.01:
            steps = step + 1
            break
            
        # Backward pass
        dZ2 = (A2 - Y) * sigmoid_derivative(Z2)
        dW2 = np.outer(dZ2, A1)
        db2 = dZ2
        
        dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(Z1)
        dW1 = np.outer(dZ1, X)
        db1 = dZ1
        
        # Update weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        
        steps = step + 1
    
    # Final forward pass for output
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    output = sigmoid(np.dot(W2, A1) + b2)

    print("\nFinal Output:", output)
    print("Binary Output:", (output > 0.5).astype(int))
    print("\nFinal Weights and Biases:")
    print("W1 (Input to Hidden Layer):\n", W1)
    print("b1 (Hidden Layer Biases):\n", b1)
    print("W2 (Hidden to Output Layer):\n", W2)
    print("b2 (Output Layer Biases):\n", b2)
    print("\nNumber of Steps:", steps)

if __name__ == "__main__":
    main()
