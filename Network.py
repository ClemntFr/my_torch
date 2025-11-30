import json, os
import numpy as np
from ActivationFunctions import ActivationFunctionFactory, softmax, sigmoid

def is_one_hot(y : np.ndarray) -> bool:
        """
        Check if the target vector is one-hot encoded.
        
        Parameters
        ----------
        y : np.ndarray
            Target vector.

        Preconditions
        -------------
        Assumes it is called within the context of a training method,
        so y has the correct shape for multi-class classification.

        Returns
        -------
        bool
            True if y is one-hot encoded, False otherwise.
        """
        return np.sum(y) == 1 and np.all((y == 0) | (y == 1))

class NeuralNetwork:
    """
        Standard feedforward neural network with multiple hidden layers.
        Uses softmax activation for output layer, multi-class cross-entropy as the loss function
        and supports different activation functions for hidden layers (defaults to ReLU).

        Member variables
        ----------------
        layers : list[int]
            Number of neurons by layer including the input and output layers
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
    def __init__(self, layers : list[int], activation="relu", weights=None, biases=None) -> None:
        """
        Initialize the weights, biases and activation functions of the network.
        Weights are initialized randomly using HE initialization and biases are 
        initialized to zero unless predefined weights and biases are given.
        
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

        self.layers = layers
        
        self.w = [np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1]) for i in range(1, len(layers))]
        self.b = [np.zeros((layers[i], 1)) for i in range(1, len(layers))]
        if weights is not None:
            self.w = weights
        if biases is not None:
            self.b = biases

        self.activation = ActivationFunctionFactory.get_activation(activation)
        # Both softmax with cross-entropy and sigmoid with binary cross-entropy give the same error in the last layer
        self.multi_class_classification = layers[-1] > 1

        if self.multi_class_classification:
            self.last_layer_activation = softmax
        else:
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

        for w, b in zip(self.w[:-1], self.b[:-1]):
            a = self.activation(np.dot(w, a) + b)

        zL = np.dot(self.w[-1], a) + self.b[-1]
        return self.last_layer_activation(zL)
    
    def evaluate(self, data: list[tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Evaluate the network's performance on the given data.
        
        Parameters
        ----------
        data : list[tuple[np.ndarray, np.ndarray]]
            Test data as a list of (input, target) tuples.

        Preconditions
        -------------
        data is not empty
            There must be test data provided.
        data[i][0].shape == (n, 1) where n is the number of input neurons
            Each input must match the input layer size.
        data[i][1].shape == (m, 1) where m is the number of output neurons
            Each target must match the output layer size.
        data[i][1] contains valid target values (e.g., one-hot encoded for multi-class or binary for binary classification)

        Returns
        -------
        float
            Accuracy of the network on the test data.
        """
        assert len(data) > 0, "Test data must not be empty."
        assert all(x.shape == (self.layers[0], 1) for x, _ in data), "Input shape does not match network input layer size."
        assert all(y.shape == (self.layers[-1], 1) for _, y in data), "Target shape does not match network output layer size."
        if self.multi_class_classification:
            assert all(is_one_hot(y) for _, y in data), "Targets must be one-hot encoded for multi-class classification."
        else:
            assert all(y[0, 0] in {0, 1} for _, y in data), "Targets must be binary (0 or 1) for binary classification."

        correct_predictions = 0
        for x, y in data:
            output = self.feedforward(x)
            predicted_label = np.argmax(output)
            true_label = np.argmax(y)
            if predicted_label == true_label:
                correct_predictions += 1
        return correct_predictions / len(data)
    
    def SGDTrain(self, data: list[tuple[np.ndarray, np.ndarray]], batch_size = 16, epochs = 1000, eta = 0.001):
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
        assert all(x.shape == (self.layers[0], 1) for x, _ in data), "Input shape does not match network input layer size."
        assert all(y.shape == (self.layers[-1], 1) for _, y in data), "Target shape does not match network output layer size."
        if self.multi_class_classification:
            assert all(is_one_hot(y) for _, y in data), "Targets must be one-hot encoded for multi-class classification."
        else:
            assert all(y[0, 0] in {0, 1} for _, y in data), "Targets must be binary (0 or 1) for binary classification."
        assert batch_size > 0, "Batch size must be positive."
        assert epochs > 0, "Number of epochs must be positive."
        assert eta > 0, "Learning rate must be positive."

        for _ in range(epochs):
            np.random.shuffle(data)
            batches = [data[k:k + batch_size] for k in range(0, len(data), batch_size)]
            for batch in batches:
                grad_w, grad_b = self.compute_mini_batch_gradients(batch)

                # Update weights and biases
                self.w = [w - eta * nw for w, nw in zip(self.w, grad_w)]
                self.b = [b - eta * nb for b, nb in zip(self.b, grad_b)]

    def RMSPropTrain(self, data: list[tuple[np.ndarray, np.ndarray]], batch_size=16, epochs=1000, eta=0.001, beta=0.9, epsilon=1e-8):
        """
        Train the network using the RMSProp algorithm with mini-batching.
        
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
        beta : float
            Decay rate
        epsilon : float
            Numerical stability constant

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
        beta > 0
            Decay rate must be positive
        epsilon > 0
            Numerical stability constant must be positive (ideally very small)

        Returns
        -------
        Nothing. Updates weights and biases in place.
        """
        assert len(data) > 0, "Training data must not be empty."
        assert all(x.shape == (self.layers[0], 1) for x, _ in data), "Input shape does not match network input layer size."
        assert all(y.shape == (self.layers[-1], 1) for _, y in data), "Target shape does not match network output layer size."
        if self.multi_class_classification:
            assert all(is_one_hot(y) for _, y in data), "Targets must be one-hot encoded for multi-class classification."
        else:
            assert all(y[0, 0] in {0, 1} for _, y in data), "Targets must be binary (0 or 1) for binary classification."
        assert batch_size > 0, "Batch size must be positive."
        assert epochs > 0, "Number of epochs must be positive."
        assert eta > 0, "Learning rate must be positive."
        assert beta > 0, "Decay rate must be positive."
        assert epsilon > 0, "Numerical stability constant must be positive."

        vw = [np.zeros_like(w) for w in self.w]
        vb = [np.zeros_like(b) for b in self.b]
        for _ in range(epochs):
            np.random.shuffle(data)
            batches = [data[k:k+batch_size] for k in range(0, len(data), batch_size)]
            for batch in batches:
                grad_w, grad_b = self.compute_mini_batch_gradients(batch)
                
                for i, (gw, gb) in enumerate(zip(grad_w, grad_b)):
                    vw[i] = beta*vw[i] + (1-beta) * (gw ** 2)
                    vb[i] = beta*vb[i] + (1-beta) * (gb ** 2)
                    self.w[i] -= eta / (epsilon + np.sqrt(vw[i])) * gw
                    self.b[i] -= eta / (epsilon + np.sqrt(vb[i])) * gb

    def AdamWTrain(self, data: list[tuple[np.ndarray, np.ndarray]], batch_size=16, epochs=1000, eta=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, l=0.01):
        """
        Train the network using adaptive moment estimation with decoupled weight
        decay (AdamW) and mini-batching.
        
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
        beta1 : float
            First moment decay
        beta2 : float
            Second moment decay
        epsilon : float
            Numerical stability constant
        l : float
            weight decay

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
        beta1 > 0 and beta2 > 0
            First and second moment decay must be positive
        epsilon > 0
            Numerical stability constant must be positive (ideally very small)
        l >= 0
            Weight decay must be positive

        Returns
        -------
        Nothing. Updates weights and biases in place.
        """
        assert len(data) > 0, "Training data must not be empty."
        assert all(x.shape == (self.layers[0], 1) for x, _ in data), "Input shape does not match network input layer size."
        assert all(y.shape == (self.layers[-1], 1) for _, y in data), "Target shape does not match network output layer size."
        if self.multi_class_classification:
            assert all(is_one_hot(y) for _, y in data), "Targets must be one-hot encoded for multi-class classification."
        else:
            assert all(y[0, 0] in {0, 1} for _, y in data), "Targets must be binary (0 or 1) for binary classification."
        assert batch_size > 0, "Batch size must be positive."
        assert epochs > 0, "Number of epochs must be positive."
        assert eta > 0, "Learning rate must be positive."
        assert beta1 > 0, "First moment decay must be positive."
        assert beta2 > 0, "Second moment decay must be positive."
        assert epsilon > 0, "Numerical stability constant must be positive."
        assert l >= 0, "Weight decay must be positive"

        vw = [np.zeros_like(w) for w in self.w]
        mw = [np.zeros_like(w) for w in self.w]
        vb = [np.zeros_like(b) for b in self.b]
        mb = [np.zeros_like(b) for b in self.b]

        for t in range(epochs):
            np.random.shuffle(data)
            batches = [data[k:k+batch_size] for k in range(0, len(data), batch_size)]
            for batch in batches:
                grad_w, grad_b = self.compute_mini_batch_gradients(batch)
                
                for i, (gw, gb) in enumerate(zip(grad_w, grad_b)):
                    vw[i] = beta2*vw[i] + (1-beta2) * (gw ** 2)
                    vb[i] = beta2*vb[i] + (1-beta2) * (gb ** 2)
                    mw[i] = beta1*mw[i] + (1-beta1) * gw
                    mb[i] = beta1*mb[i] + (1-beta1) * gb

                    vw_hat = vw[i] / (1 - beta2 ** (t+1))
                    vb_hat = vb[i] / (1 - beta2 ** (t+1))
                    mw_hat = mw[i] / (1 - beta1 ** (t+1))
                    mb_hat = mb[i] / (1 - beta1 ** (t+1))

                    self.w[i] -= eta * l * self.w[i]

                    self.w[i] -= eta / (epsilon + np.sqrt(vw_hat)) * mw_hat
                    self.b[i] -= eta / (epsilon + np.sqrt(vb_hat)) * mb_hat

    def compute_mini_batch_gradients(self, batch: list[tuple[np.ndarray, np.ndarray]]):
        """
        Compute the loss function's mean gradients relative to the weights and
        biases of the network over single mini batch.

        Parameters
        ----------
        batch : list[tuple[np.array, np.array]]
            A mini-batch of training data as a list of (input, target) tuples.

        Preconditions
        -------------
        Assumes this function is called within the context of a training method,
        so the batch is non-empty and inputs/targets have correct shapes / encoding.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Gradient of the loss function relative to weights and biases
        """
        grad_w = [np.zeros_like(w) for w in self.w]
        grad_b = [np.zeros_like(b) for b in self.b]
        
        for x, y in batch:
            dgrad_w, dgrad_b = self.backpropagate(x, y)
            grad_w = [gw + dgw for gw, dgw in zip(grad_w, dgrad_w)]
            grad_b = [gb + dgb for gb, dgb in zip(grad_b, dgrad_b)]

        grad_w = [gw/len(batch) for gw in grad_w]
        grad_b = [gb/len(batch) for gb in grad_b]
        
        return grad_w, grad_b

    def get_activations_and_logits(self, a: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compute and return all activations and weighted inputs (logits) for a given input.
        
        Parameters
        ----------
        a : np.ndarray
            Input features.
        
        Preconditions
        -------------
        Assumes it is called within the context of the backpropagate method,
        so a has the correct shape for the input layer.

        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            Activations and weighted inputs for each layer.
        """
        activations = [a]
        weighted_inputs = []
        for w, b in zip(self.w[:-1], self.b[:-1]):
            z = np.dot(w, a) + b
            weighted_inputs.append(z)
            a = self.activation(z)
            activations.append(a)

        zL = np.dot(self.w[-1], a) + self.b[-1]
        weighted_inputs.append(zL)
        aL = self.last_layer_activation(zL)
        activations.append(aL)
        
        return activations, weighted_inputs

    def backpropagate(self, x : np.ndarray, y : np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Compute the gradient of the cost function with respect to
        weights and biases for a single (input, target) pair using backpropagation.
        
        Parameters
        ----------
        y : float
            True binary label (0 or 1).
        x : np.ndarray
            Input features.
        
        Returns
        -------
        tuple[list[np.ndarray], list[np.ndarray]]
            Gradients with respect to weights and biases.
        """
        a, z = self.get_activations_and_logits(x)

        grad_w = [np.zeros_like(w) for w in self.w]
        grad_b = [np.zeros_like(b) for b in self.b]

        # Output layer
        error = a[-1] - y  # with softmax for last layer activation and multi-class cross-entropy for loss
        grad_w[-1] = np.dot(error, a[-2].transpose())
        grad_b[-1] = error

        for l in range(2, len(self.w) + 1):
            error = np.dot(self.w[-l+1].T, error) * self.activation.derivative(z[-l])
            grad_w[-l] = np.dot(error, a[-l-1].transpose())
            grad_b[-l] = error
        
        return grad_w, grad_b
    
class NetworkManager:
    def __init__(self):
        pass

    @staticmethod
    def save_network(net : NeuralNetwork, filepath : str):
        """
        Saves the neural network to a .mnet metafile containing
        every data about the network except the weights and biases which
        are stored to a .npz file in the same directory
        """
        path, _ = os.path.splitext(filepath)

        metapath = path + ".mnet"
        wb_filepath = path + ".npz"
        net_dict = {
            "layers" : net.layers, 
            "activation" : net.activation.to_str(), 
            "wb_filepath" : wb_filepath
        }
        json_str = json.dumps(net_dict, indent=4)

        # Saving hyperparametters
        with open(metapath, 'w') as network_metafile:
            network_metafile.write(json_str)

        # Saving weights and biases
        np.savez(wb_filepath,
         **{f"W{i}": w for i, w in enumerate(net.w)},
         **{f"B{i}": b for i, b in enumerate(net.b)})
        
    @staticmethod
    def load_network(metapath : str):
        # Loading metadata
        with open(metapath, 'r') as network_metafile:
            network_data = json.load(network_metafile)

        # Initializing new network
        network = NeuralNetwork(network_data["layers"], network_data["activation"])

        # Setting up weights and biases
        params = np.load(network_data["wb_filepath"])
        network.w = [params[f"W{i}"] for i in range(len(network.w))]
        network.b = [params[f"B{i}"] for i in range(len(network.b))]

        return network
    
    @staticmethod
    def randomize_network(network : NeuralNetwork):
        """
        Re-initializes the weights and biases of the network
        Weights are initialized randomly using He initialization
        and biases are set to zero.
        """
        for i in range(1, len(network.layers)):
            network.w = np.random.randn(network.layers[i], network.layers[i-1]) * np.sqrt(2/network.layers[i-1])
            network.b = np.zeros((network.layers[i], 1))

        
    
if __name__ == "__main__":
    # Simple test case
    nn = NeuralNetwork(layers=[2, 5, 1], activation="sigmoid")
    
    # Dummy data: XOR problem
    data = [
        (np.array([[0], [0]]), np.array([[0]])),
        (np.array([[0], [1]]), np.array([[1]])),
        (np.array([[1], [0]]), np.array([[1]])),
        (np.array([[1], [1]]), np.array([[0]]))
    ]
    X = np.array([x for x, _ in data])
    Y = np.array([y for _, y in data])
    X = (X - X.mean(axis=0)) / X.std(axis=0)  # Normalize inputs
    data = list(zip([X[i].reshape(-1, 1) for i in range(X.shape[0])],
                    [Y[i].reshape(-1, 1) for i in range(Y.shape[0])]))

    # Before training
    for x, y in data:
        output = nn.feedforward(x)
        print(f"Input: {x.ravel()}, Target: {y.ravel()}, Output before training: {output.ravel()}")
    
    print(nn.get_activations_and_logits(np.array([[1], [-1]])))
    # Training
    print("Training...")
    nn.AdamWTrain(data, batch_size=2, epochs=1000, eta=0.1, l=0.01, beta2=0.9)

    # After training
    for x, y in data:
        output = nn.feedforward(x)
        print(f"Input: {x.ravel()}, Target: {y.ravel()}, Output after training: {output.ravel()}")

    path = "./SavedModels/XOR/xor.mnet"
    print("Saving to: " + path)
    NetworkManager.save_network(nn, path)

    print("Loading back...")
    newnet = NetworkManager.load_network(path)

    for x, y in data:
        output = nn.feedforward(x)
        print(f"Input: {x.ravel()}, Target: {y.ravel()}, Output of the loaded network: {output.ravel()}")
    