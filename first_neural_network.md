# First Neural Network Training Issue Documentation

## Problem Analysis #1

In the provided neural network implementation, there is a significant issue with the **backpropagation** function, particularly in the calculation of the bias gradients (`db1` and `db2`). The current code calculates the sum of gradients along the wrong axis, which can hinder the learning process.

### Specific Issues

- **Incorrect Axis for Summation**: The gradients for biases (`db1` and `db2`) are being summed along axis 1 instead of axis 0. This prevents proper averaging of the gradients across all training examples, leading to stagnant training and low accuracy.

## Proposed Solution

To resolve the issue, the following changes should be made to the `back_prop` function:

### Modifications to Backpropagation Function

```python
def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)  # Changed axis to 0 and keepdims to True
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)  # Changed axis to 0 and keepdims to True
    return dW1, db1, dW2, db2
```
## Reasoning Behind the Changes

1. **Correct Averaging**: By summing along axis 0, we ensure that we average the gradients for each bias across all training examples. This results in a more stable update to the model parameters.
  
2. **Avoid Broadcasting Errors**: The use of `keepdims=True` maintains the dimensionality of the output, preventing any potential broadcasting issues during parameter updates.

## Complete Code Implementation

Here’s the full implementation with the necessary adjustments:

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import kaggle

# Download dataset
kaggle.api.competition_download_file('digit-recognizer', 'train.csv')
data = pd.read_csv('/home/garv/Data-Science/train.csv')
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

# Development and training datasets
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255
_, m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2 

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z), axis=0)

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
    
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)  # Fixed
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)  # Fixed
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if (i % 10 == 0):
            print('Iteration: ', i)
            print('Accuracy: ', get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2

# Start training the model
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 100, 0.75)
```
## Problem Analysis #2

1. **Stagnant Training**:
   - The model's accuracy remains constant at 10%, indicating that it's likely not learning effectively.

2. **Potential Causes**:
   - **Weight Initialization**: Random values initialized around -0.5 may lead to poor gradient flow and slow convergence.
   - **High Learning Rate**: A learning rate set too high (e.g., 0.5) can cause the model to oscillate or overshoot optimal weights, preventing effective learning.
   - **Activation Function Choice**: Using only ReLU may lead to dead neurons, especially if initialized weights are not optimal.

## Solutions

1. **Weight Initialization**:
   - Change the initialization of weights to a smaller range centered around 0, which helps prevent saturation of activation functions. Here's an example:
     ```python
     def init_params():
         W1 = np.random.randn(10, 784) * 0.01  # Smaller range
         b1 = np.zeros((10, 1))  # Initialize biases to 0
         W2 = np.random.randn(10, 10) * 0.01  # Smaller range
         b2 = np.zeros((10, 1))  # Initialize biases to 0
         return W1, b1, W2, b2
     ```

2. **Adjust Learning Rate**:
   - Reduce the learning rate to allow for more gradual updates. For example:
     ```python
     W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 100, 0.1)  # Reduced alpha to 0.1
     ```

3. **Experiment with Activation Functions**:
   - While ReLU is effective, testing other activation functions like sigmoid or tanh may provide better results for certain datasets, particularly in hidden layers.

## Reasoning Behind the Changes

- **Weight Initialization**: Initializing weights with smaller values prevents activation functions from saturating early in training, facilitating better gradient flow and more effective learning.

- **Learning Rate**: A lower learning rate enables smoother convergence towards the optimal solution without overshooting, which can destabilize training.

- **Activation Functions**: Different activation functions can influence learning dynamics; experimenting with various functions allows you to find the most effective configuration for your data and model architecture.

## Conclusion

By implementing these modifications, you should see an improvement in training dynamics and an increase in model accuracy. Always monitor the performance and be ready to adjust parameters based on observed outcomes during training.

For more information on weight initialization and learning rates, you can explore these resources:
- [Understanding Weight Initialization](https://towardsdatascience.com/understanding-weight-initialization-in-deep-learning-5f99464b1c45)
- [Learning Rate Scheduling](https://machinelearningmastery.com/learning-rate-schedules-for-deep-learning-neural-networks/)

