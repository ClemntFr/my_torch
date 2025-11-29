from ActivationFunctions import ActivationFunctionFactory
import numpy as np
import matplotlib.pyplot as plt

def softmax(z : np.ndarray) -> np.ndarray:
    """
    Compute the softmax of a vector or a batch of vectors in a numerically stable way.
    
    Parameters
    ----------
    z: np.ndarray
        Input array. shape (n,), (n,1), or (n,m)
    
    Returns
    -------
    np.ndarray
        Softmax probabilities applied column-wise with the same shape as x.
    """
    # Convert 1D vectors into column vectors internally
    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # Numerical stability: subtract max of each column
    z_shifted = z - np.max(z, axis=0, keepdims=True)

    exp_z = np.exp(z_shifted)
    softmax_vals = exp_z / np.sum(exp_z, axis=0, keepdims=True)

    return softmax_vals

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y_hat, y):
    """
    Compute the binary cross-entropy loss.
    
    Parameters
    ----------
    y : float
        True binary label (0 or 1).
    y_hat : float
        Predicted probability (between 0 and 1).
    
    Returns
    -------
    float
        Binary cross-entropy loss.
    """
    # Clip predictions to avoid log(0)
    y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

def mean_squared_error(y_hat, y):
    """
    Compute the mean squared error loss.
    
    Parameters
    ----------
    y : float
        True binary label (0 or 1).
    y_hat : float
        Predicted probability (between 0 and 1).
    
    Returns
    -------
    float
        Mean squared error loss.
    """
    return np.mean((y_hat - y) ** 2)

class Network1:
    """
        - [Inputs] Two inputs
        - [Hidden Layers] None
        - [Hidden Layers activation] N/A
        - [Outputs] One output ranging from 0 to 1 (probability of class 1)
        - [Loss Function] Binary Cross-Entropy Loss
        - [Training] Stochastic Gradient Descent (SGD)
        - [Activation Function] Sigmoid activation function for output layer
    """
    def __init__(self, weights=None, bias=None):
        # Initialize weights and biases
        self.w = np.random.randn(2)   # One output neuron, two input features
        self.b = np.random.randn()     # One bias for the output neuron
        if weights is not None:
            self.w = weights
        if bias is not None:
            self.b = bias

    def feedforward(self, a: np.array) -> np.array:
        z = np.dot(self.w, a) + self.b
        return sigmoid(z)
    
    def train(self, data: list[tuple[np.array, float]], epochs: int, eta: float):
        for _ in range(epochs):
            np.random.shuffle(data)
            for x, y in data:
                self.update_weights(x, y, eta)
    
    def update_weights(self, x, y, eta):
        # Perform a single step of stochastic gradient descent
        y_hat = self.feedforward(x)
        error = y_hat - y  # with sigmoid and binary cross-entropy
        
        # Gradient calculation
        grad_w = error * x
        grad_b = error
        
        # Update weights and bias
        self.w -= eta * grad_w
        self.b -= eta * grad_b

class Network11:
    """
        - [Inputs] Two inputs
        - [Hidden Layers] None
        - [Hidden Layers activation] N/A
        - [Outputs] One output ranging from 0 to 1 (probability of class 1)
        - [Loss Function] Mean Squared Error Loss
        - [Training] Stochastic Gradient Descent (SGD)
        - [Activation Function] Sigmoid activation function for output layer
    """
    def __init__(self, weights=None, bias=None):
        # Initialize weights and biases
        self.w = np.random.randn(2)   # One output neuron, two input features
        self.b = np.random.randn()     # One bias for the output neuron
        if weights is not None:
            self.w = weights
        if bias is not None:
            self.b = bias

    def feedforward(self, a: np.array) -> np.array:
        z = np.dot(self.w, a) + self.b
        return sigmoid(z)
    
    def train(self, data: list[tuple[np.array, float]], epochs: int, eta: float):
        for _ in range(epochs):
            np.random.shuffle(data)
            for x, y in data:
                self.update_weights(x, y, eta)

    def update_weights(self, x, y, eta):
        # Perform a single step of stochastic gradient descent
        y_hat = self.feedforward(x)
        error = y_hat * (1 - y_hat) * (y_hat - y)  # with sigmoid and mean squared error
        
        # Gradient calculation
        grad_w = error * x
        grad_b = error
        
        # Update weights and bias
        self.w -= eta * grad_w
        self.b -= eta * grad_b

class Network2:
    """
        - [Inputs] Two inputs
        - [Hidden Layers] One hidden layer with one neuron
        - [Hidden Layers activation] Sigmoid (default) or ReLu activation function
        - [Outputs] One output ranging from 0 to 1 (probability of class 1)
        - [Loss Function] Binary Cross-Entropy Loss
        - [Training] Stochastic Gradient Descent (SGD)
        - [Activation Function] Sigmoid activation function for output layer
    """
    def __init__(self, activation="sigmoid", weights=None, biases=None):
        # Initialize weights and biases
        self.w = [np.random.randn(1, 2), np.random.randn(1)]   # two weights for the hidden neuron and one for the output neuron
        self.b = np.random.randn(2, 1)     # One bias for the hidden neuron and one for the output neuron
        if weights is not None:
            self.w = weights
        if biases is not None:
            self.b = biases
        self.activation = ActivationFunctionFactory.get_activation(activation)

    def feedforward(self, a: np.array) -> np.array:
        z = 0
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            a = self.activation(z)
        return sigmoid(z)
    
    def train(self, data: list[tuple[np.array, float]], epochs: int, eta: float):
        for _ in range(epochs):
            np.random.shuffle(data)
            for x, y in data:
                self.update_weights(x, y, eta)

    def get_activations_and_weighted_inputs(self, a: np.array) -> tuple[list[np.array], list[np.array]]:
        activations = [a]
        weighted_inputs = []
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            weighted_inputs.append(z)
            a = self.activation(z)
            activations.append(a)
        return activations, weighted_inputs

    def update_weights(self, x, y, eta):
        """
        Perform a single step of stochastic gradient descent
        
        Parameters
        ----------
        y : float
            True binary label (0 or 1).
        x : np.ndarray
            Input features.
        
        Returns
        -------
        Nothing. Updates weights and biases in place.
        """
        a, z = self.get_activations_and_weighted_inputs(x)

        grad_w = [np.zeros_like(w) for w in self.w]
        grad_b = [np.zeros_like(b) for b in self.b]

        # Output layer
        error = a[-1] - y  # with sigmoid for last layer activation and binary cross-entropy
        grad_w[-1] = np.dot(error.reshape(-1, 1), a[-2].reshape(1, -1))
        grad_b[-1] = error

        # Hidden layer
        error = np.dot(self.w[-1].T, error) * self.activation.derivative(z[0])
        grad_w[0] = np.dot(error.reshape(-1, 1), x.reshape(1, -1))
        grad_b[0] = error
        
        # Update weights and biases        
        self.w = [w - eta * g_w for w, g_w in zip(self.w, grad_w)]
        self.b = [b - eta * g_b for b, g_b in zip(self.b, grad_b)]

class Network3:
    """
        - [Inputs] Two inputs
        - [Hidden Layers] One hidden layer with n neurons
        - [Hidden Layers activation] Sigmoid (default) or ReLu activation function
        - [Outputs] One output ranging from 0 to 1 (probability of class 1)
        - [Loss Function] Binary Cross-Entropy Loss
        - [Training] Stochastic Gradient Descent (SGD)
        - [Activation Function] Sigmoid activation function for output layer
    """
    def __init__(self, n = 1, activation="sigmoid", weights=None, biases=None):
        # Initialize weights and biases
        self.w = [np.random.randn(n, 2), np.random.randn(1, n)]
        self.b = [np.random.randn(n, 1), np.random.randn(1, 1)] 
        if weights is not None:
            self.w = weights
        if biases is not None:
            self.b = biases
        self.activation = ActivationFunctionFactory.get_activation(activation)

    def feedforward(self, a: np.array) -> np.array:
        a = a.reshape(-1, 1)
        z = 0
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            a = self.activation(z)
        return sigmoid(z)
    
    def train(self, data: list[tuple[np.array, float]], epochs: int, eta: float):
        for _ in range(epochs):
            np.random.shuffle(data)
            for x, y in data:
                self.update_weights(x, y, eta)

    def get_activations_and_weighted_inputs(self, a: np.array) -> tuple[list[np.array], list[np.array]]:
        a = a.reshape(-1, 1)
        activations = [a]
        weighted_inputs = []
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            weighted_inputs.append(z)
            a = self.activation(z)
            activations.append(a)
        return activations, weighted_inputs

    def update_weights(self, x, y, eta):
        """
        Perform a single step of stochastic gradient descent
        
        Parameters
        ----------
        y : float
            True binary label (0 or 1).
        x : np.ndarray
            Input features.
        
        Returns
        -------
        Nothing. Updates weights and biases in place.
        """
        a, z = self.get_activations_and_weighted_inputs(x)

        grad_w = [np.zeros_like(w) for w in self.w]
        grad_b = [np.zeros_like(b) for b in self.b]

        # Output layer
        error = a[-1] - y  # with sigmoid for last layer activation and binary cross-entropy
        grad_w[-1] = np.dot(error.reshape(-1, 1), a[-2].reshape(1, -1))
        grad_b[-1] = error

        # Hidden layer
        error = np.dot(self.w[-1].T, error) * self.activation.derivative(z[0])
        grad_w[0] = np.dot(error.reshape(-1, 1), x.reshape(1, -1))
        grad_b[0] = error
        
        # Update weights and biases        
        self.w = [w - eta * g_w for w, g_w in zip(self.w, grad_w)]
        self.b = [b - eta * g_b for b, g_b in zip(self.b, grad_b)]

class Network4:
    """
        - [Inputs] 'input_nbr' inputs
        - [Hidden Layers] One hidden layer with 'n' neurons
        - [Hidden Layers activation] Sigmoid (default) or ReLu activation function
        - [Outputs] One output ranging from 0 to 1 (probability of class 1)
        - [Loss Function] Binary Cross-Entropy Loss
        - [Training] Stochastic Gradient Descent (SGD)
        - [Activation Function] Sigmoid activation function for output layer
    """
    def __init__(self, input_nbr = 2, n = 1, activation="sigmoid", weights=None, biases=None):
        # Initialize weights and biases
        self.w = [np.random.randn(n, input_nbr), np.random.randn(1, n)]
        self.b = [np.random.randn(n, 1), np.random.randn(1, 1)] 
        if weights is not None:
            self.w = weights
        if biases is not None:
            self.b = biases
        self.activation = ActivationFunctionFactory.get_activation(activation)

    def feedforward(self, a: np.array) -> np.array:
        a = a.reshape(-1, 1)
        z = 0
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            a = self.activation(z)
        return sigmoid(z)
    
    def train(self, data: list[tuple[np.array, float]], epochs: int, eta: float):
        for _ in range(epochs):
            np.random.shuffle(data)
            for x, y in data:
                self.update_weights(x, y, eta)

    def get_activations_and_weighted_inputs(self, a: np.array) -> tuple[list[np.array], list[np.array]]:
        a = a.reshape(-1, 1)
        activations = [a]
        weighted_inputs = []
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            weighted_inputs.append(z)
            a = self.activation(z)
            activations.append(a)
        return activations, weighted_inputs

    def update_weights(self, x, y, eta):
        """
        Perform a single step of stochastic gradient descent
        
        Parameters
        ----------
        y : float
            True binary label (0 or 1).
        x : np.ndarray
            Input features.
        
        Returns
        -------
        Nothing. Updates weights and biases in place.
        """
        a, z = self.get_activations_and_weighted_inputs(x)

        grad_w = [np.zeros_like(w) for w in self.w]
        grad_b = [np.zeros_like(b) for b in self.b]

        # Output layer
        error = a[-1] - y  # with sigmoid for last layer activation and binary cross-entropy
        grad_w[-1] = np.dot(error.reshape(-1, 1), a[-2].reshape(1, -1))
        grad_b[-1] = error

        # Hidden layer
        error = np.dot(self.w[-1].T, error) * self.activation.derivative(z[0])
        grad_w[0] = np.dot(error.reshape(-1, 1), x.reshape(1, -1))
        grad_b[0] = error
        
        # Update weights and biases        
        self.w = [w - eta * g_w for w, g_w in zip(self.w, grad_w)]
        self.b = [b - eta * g_b for b, g_b in zip(self.b, grad_b)]

class NeuralNetwork:
    """
        Standard feedforward neural network with multiple hidden layers.
        Uses softmax activation for output layer, multi-class cross-entropy as the loss function
        and supports different activation functions for hidden layers.

        Member variables
        ----------------
        input_size : int.
            Number of input features.
        output_size : int.
            Number of output classes.
        w : list[np.ndarray].
            Weights for each layer stored as 2D matrices.
        b : list[np.ndarray].
            Biases for each layer stored as 2D column vectors.
        activation : ActivationFunction.
            Activation function for hidden layers.
        last_layer_activation : function.
            Activation function for output layer (softmax or sigmoid).
        multi_class_classification : bool.
            True if the network is used for multi-class classification, False for binary classification.
    """
    def __init__(self, layers : list[int], activation="sigmoid", weights=None, biases=None) -> None:
        """
        Initialize the weights, biases and activation functions of the network.
        Weights and biases are initialized randomly using a normal distribution
        or can be provided as arguments.
        
        Parameters
        ----------
        layers: list[int]
            Number of neurons in each layer including input and output layers.
        activation: str
            Activation function to use in hidden layers ("sigmoid" or "relu").
        weights: list[np.ndarray], optional
            Predefined weights for each layer stored as 2D matrices
        biases: list[np.ndarray], optional
            Predefined biases for each layer stored as 2D column vectors
        
        Preconditions
        -------------
        len(layers) >= 2:
            The network must have at least an input and an output layer.
        all(l > 0 for l in layers):
            The number of neurons in each layer must be strictly positive.
        activation in {"sigmoid", "relu", "identity"} :
            The activation function must be one of the supported types.
            (Checked in ActivationFunctionFactory.get_activation)
        """
        assert len(layers) >= 2, "Network must have at least input and output layers."
        assert all(l > 0 for l in layers), "Number of neurons in each layer must be positive."""

        self.input_size = layers[0]
        self.output_size = layers[-1]
        
        self.w = [np.random.randn(layers[i], layers[i-1]) for i in range(1, len(layers))]
        self.b = [np.random.randn(layers[i], 1) for i in range(1, len(layers))]
        if weights is not None:
            self.w = weights
        if biases is not None:
            self.b = biases

        self.activation = ActivationFunctionFactory.get_activation(activation)
        # Both softmax with cross-entropy and sigmoid with binary cross-entropy give the same error in the last layer
        self.multi_class_classification = layers[-1] > 1
        self.last_layer_activation = softmax
        if not self.multi_class_classification:
            self.last_layer_activation = sigmoid

    def feedforward(self, a: np.ndarray) -> np.ndarray:
        """
        Compute the output of the network for a given input.

        Parameters
        ----------
        a : np.ndarray
            Input features.

        Preconditions
        -------------
        a.shape == (n, 1) where n is the number of input neurons.
        """
        assert a.shape == (self.w[0].shape[1], 1), f"Input shape {a.shape} does not match expected shape {(self.w[0].shape[1], 1)}."

        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            a = self.activation(z)
        return self.last_layer_activation(z)
    
    def _is_one_hot(self, y : np.ndarray) -> bool:
        """
        Check if the target vector is one-hot encoded.
        
        Parameters
        ----------
        y : np.ndarray
            Target vector.

        Preconditions
        -------------
        Assumes it is called within the context of the train method,
        so y has the correct shape for multi-class classification.

        Returns
        -------
        bool
            True if y is one-hot encoded, False otherwise.
        """
        return np.sum(y) == 1 and np.all((y == 0) | (y == 1))
    
    
    def train(self, data: list[tuple[np.array, np.array]], batch_size : int = 16, epochs: int = 1000, eta: float = 0.1):
        """
        Train the network using mini-batch stochastic gradient descent.
        
        Parameters
        ----------
        data : list[tuple[np.array, np.array]]
            Training data as a list of (input, target) tuples.
        batch_size : int
            Size of each mini-batch.
        epochs : int
            Number of epochs to train.
        eta : float
            Learning rate.

        Preconditions
        -------------
        data is not empty
            There must be training data provided.
        data[i][0].shape == (n, 1) where n is the number of input neurons
            Each input must match the input layer size.
        data[i][1].shape == (m, 1) where m is the number of output neurons
            Each target must match the output layer size.
        data[i][1] contains valid target values (e.g., one-hot encoded for multi-class or binary for binary classification)
        batch_size > 0
            Batch size must be positive.
        epochs > 0
            Number of epochs must be positive.
        eta > 0
            Learning rate must be positive.

        Returns
        -------
        Nothing. Updates weights and biases in place.
        """
        assert len(data) > 0, "Training data must not be empty."
        assert all(x.shape == (self.input_size, 1) for x, _ in data), "Input shape does not match network input layer size."
        assert all(y.shape == (self.w[-1].shape[0], 1) for _, y in data), "Target shape does not match network output layer size."
        if self.multi_class_classification:
            assert all(self._is_one_hot(y) for _, y in data), "Targets must be one-hot encoded for multi-class classification."
        else:
            assert all(y[0, 0] in {0, 1} for _, y in data), "Targets must be binary (0 or 1) for binary classification."
        assert batch_size > 0, "Batch size must be positive."
        assert epochs > 0, "Number of epochs must be positive."
        assert eta > 0, "Learning rate must be positive."

        for _ in range(epochs):
            np.random.shuffle(data)
            batches = [data[k:k + batch_size] for k in range(0, len(data), batch_size)]
            for batch in batches:
                self.update_mini_batch(batch, eta)

    def update_mini_batch(self, batch: list[tuple[np.array, np.array]], eta: float):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.

        Parameters
        ----------
        batch : list[tuple[np.array, np.array]]
            A mini-batch of training data as a list of (input, target) tuples.
        eta : float
            Learning rate.

        Preconditions
        -------------
        Assumes this function is called within the context of the train method,
        so the batch is non-empty and inputs/targets have correct shapes / encoding.

        Returns
        -------
        Nothing. Updates weights and biases in place.
        """
        grad_w = [np.zeros_like(w) for w in self.w]
        grad_b = [np.zeros_like(b) for b in self.b]
        
        for x, y in batch:
            dgrad_w, dgrad_b = self.backpropagate(x, y)
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, dgrad_w)]
            grad_b = [gb + dgb for gb, dgb in zip(grad_b, dgrad_b)]
        
        # Update weights and biases
        self.w = [w - (eta / len(batch)) * nw for w, nw in zip(self.w, grad_w)]
        self.b = [b - (eta / len(batch)) * nb for b, nb in zip(self.b, grad_b)]

    def get_activations_and_logits(self, a: np.array) -> tuple[list[np.array], list[np.array]]:
        a = a.reshape(-1, 1)
        activations = [a]
        weighted_inputs = []
        for w, b in zip(self.w, self.b):
            z = np.dot(w, a) + b
            weighted_inputs.append(z)
            a = self.activation(z)
            activations.append(a)
        return activations, weighted_inputs

    def backpropagate(self, x, y):
        """
        Perform a single step of stochastic gradient descent
        
        Parameters
        ----------
        y : float
            True binary label (0 or 1).
        x : np.ndarray
            Input features.
        
        Returns
        -------
        Nothing. Updates weights and biases in place.
        """
        a, z = self.get_activations_and_logits(x)

        grad_w = [np.zeros_like(w) for w in self.w]
        grad_b = [np.zeros_like(b) for b in self.b]

        # Output layer
        error = a[-1] - y  # with softmax for last layer activation and multi-class cross-entropy for loss
        grad_w[-1] = np.dot(error.reshape(-1, 1), a[-2].reshape(1, -1))
        grad_b[-1] = error

        for l in range(2, len(self.w) + 1):
            error = np.dot(self.w[-l + 1].T, error) * self.activation.derivative(z[-l])
            grad_w[-l] = np.dot(error, a[-l - 1].transpose())
            grad_b[-l] = error
        
        return grad_w, grad_b
    
if __name__ == "__main__":
    
    training_data = [
        (np.array([0, 0, 0]).reshape(-1, 1), np.array([1, 0]).reshape(-1, 1)),
        (np.array([0, 0, 1]).reshape(-1, 1), np.array([1, 0]).reshape(-1, 1)),
        (np.array([0, 1, 0]).reshape(-1, 1), np.array([1, 0]).reshape(-1, 1)),
        (np.array([0, 1, 5]).reshape(-1, 1), np.array([1, 0]).reshape(-1, 1)),
        (np.array([1, 0, 0]).reshape(-1, 1), np.array([0, 1]).reshape(-1, 1)),
        (np.array([1, 0, 1]).reshape(-1, 1), np.array([0, 1]).reshape(-1, 1)),
        (np.array([1, 1, 0]).reshape(-1, 1), np.array([0, 1]).reshape(-1, 1)),
        (np.array([1, 1, 1]).reshape(-1, 1), np.array([1, 0]).reshape(-1, 1))
    ]

    net = NeuralNetwork(layers=[3, 10, 5, 2], activation="sigmoid")
    # Test the network
    for x, y in training_data:
        output = net.feedforward(x)
        print(f"Input: {x}, Predicted: {output}, True: {y}")

    print("Training...")
    net.train(training_data, batch_size=4, epochs=1000, eta=0.1)
    
    # Test the trained network
    for x, y in training_data:
        output = net.feedforward(x)
        print(f"Input: {x}, Predicted: {output}, True: {y}")
    
    