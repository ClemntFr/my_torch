import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    @abstractmethod
    def __call__(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    @abstractmethod
    def to_str(self):
        pass

class Sigmoid(ActivationFunction):
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        s = self.__call__(x)
        return s * (1 - s)
    
    def to_str(self):
        return "sigmoid"
    
class Identity(ActivationFunction):
    def __call__(self, x):
        return x
    
    def derivative(self, x):
        return 1

    def to_str(self):
        return "identity"
    
class ReLU(ActivationFunction):
    def __call__(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        # Derivative of ReLU is 1 for x > 0, else 0
        # Should ideally handle x == 0 case, but for simplicity we treat it as 0
        return (x > 0).astype(float)

    def to_str(self):
        return "relu"
    
class ActivationFunctionFactory:
    @staticmethod
    def get_activation(name: str) -> ActivationFunction:
        valid_activations = {"sigmoid" : Sigmoid(), "identity": Identity(), "relu": ReLU()}
        name = name.lower()
        assert(name in valid_activations), f"Invalid activation function name: {name}"
        
        return valid_activations[name]
    
def softmax(z : np.ndarray) -> np.ndarray:
    """
    Compute the softmax of a column vector or a batch of vectors in a numerically stable way.
    
    Parameters
    ----------
    z: np.ndarray
        Input array. shape (n,), (n,1), or (n,m)

    Preconditions
    -------------
    Assumes z 2D array where each column is a separate input vector.
    
    Returns
    -------
    np.ndarray
        Softmax probabilities applied column-wise with the same shape as x.
    """

    # Numerical stability: subtract max of each column
    z_shifted = z - np.max(z, axis=0, keepdims=True)

    exp_z = np.exp(z_shifted)
    softmax_vals = exp_z / np.sum(exp_z, axis=0, keepdims=True)

    return softmax_vals

def sigmoid(z : np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid activation function.
    
    Parameters
    ----------
    z: np.ndarray
        Input array.
    
    Returns
    -------
    np.ndarray
        Sigmoid applied element-wise with the same shape as x.
    """
    return 1 / (1 + np.exp(-z))