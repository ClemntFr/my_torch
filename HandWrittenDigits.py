import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

from Network import NeuralNetwork, NetworkManager

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)
    

def normalize(x : np.ndarray) -> np.ndarray:
    return (x - x.mean(axis=0)) / x.std(axis=0)

if __name__ == "__main__":
    data_path = "MNIST"
    mnist_dataloader = MnistDataloader(join(data_path, 'train-images.idx3-ubyte'),
                                       join(data_path, 'train-labels.idx1-ubyte'),
                                       join(data_path, 't10k-images.idx3-ubyte'),
                                       join(data_path, 't10k-labels.idx1-ubyte'))
    print("Loading database...")
    (x_train, y_train),(x_test, y_test) = mnist_dataloader.load_data()
    print("Database loaded.")
    print("Preprocessing data...")
    # One-hot encoding of labels
    y_train_one_hot = [np.zeros((10, 1)) for _ in range(len(y_train))]
    y_test_one_hot = [np.zeros((10, 1)) for _ in range(len(y_test))]
    for i, label in enumerate(y_train):
        y_train_one_hot[i][label][0] = 1
    for i, label in enumerate(y_test):
        y_test_one_hot[i][label][0] = 1


    # Convert images to numpy arrays, normalize and flatten them
    for i, x in enumerate(x_train):
        x_train[i] = normalize((np.array(x).reshape(784, 1)))
    for i, x in enumerate(x_test):
        x_test[i] = normalize((np.array(x).reshape(784, 1)))

    training_data = list(zip(x_train, y_train_one_hot))
    test_data = list(zip(x_test, y_test_one_hot))

    print("Preprocessing done.")

    # ----------- Actual Neural Network shit -----------
    if (input("Load past model ? (y/n): ") == 'y'):
        net = NetworkManager.load_network(input("Model metafile path (*.mnet): "))
    else:
        print("Creating Network...")
        layers=[784, 128, 64, 10]
        activation="ReLU"
        net = NeuralNetwork(layers, activation)

    print("Layer sizes :", net.layers)
    print("Activation function for hidden layers :", net.activation.to_str())
    print("Activation function for output layer : softmax")
    
    if (input("Train ? (y/n): ") == 'y'):
        epochs=10
        batch_size=32
        eta=0.001
        beta1=0.9
        beta2=0.999
        epsilon=1e-8
        print("Starting training with :")
        print(f"\t{len(training_data)} training samples")
        print(f"\t{batch_size} samples per batch")
        print(f"\t{epochs} epochs")
        print(f"\tlearning rate : {eta}")
        print(f"\tfirst moment decay : {beta1}")
        print(f"\tsecond moment decay : {beta2}")
        print(f"\tnumerical stability constant: {epsilon}")
        net.AdamTrain(training_data, batch_size, epochs, eta, beta1, beta2, epsilon)
    
    accuracy = net.evaluate(test_data)
    print(f"Test dataset accuracy: {accuracy * 100:.2f}%")

    if (input("Save ? (y/n): ") == 'y'):
        NetworkManager.save_network(net, input("Where to save ? "))
