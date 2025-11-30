import json
import chess
import numpy as np
import random
from Network import NeuralNetwork, NetworkManager

"""
    One hot encoding :
        [checkmate white, checkmate black, check white, check black, nothing]
"""

piece_to_index = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5
}

def fen_to_vector(fen: str):
    board = chess.Board(fen)
    
    vec = np.zeros(769, dtype=np.float32)
    
    # --- 12x64 piece planes flattened ---
    for square, piece in board.piece_map().items():
        base = 64 * piece_to_index[piece.piece_type]
        if piece.color == chess.BLACK:
            base += 6 * 64

        vec[base + square] = 1.0
    
    idx = 12 * 64  # = 768
    
    # --- Side to move (1 feature) ---
    vec[idx] = 1.0 if board.turn == chess.WHITE else -1.0
    
    return vec.reshape(-1, 1)

def load_data(json_path):
    with open(json_path, "r") as f:
        fens = json.load(f)

    return fens

def preprocess_data(fens):
    # Add every position as a np.ndarray vector with its associated one hot encoded state to the data
    # and returns
    states = ["checkmate white", "checkmate black", "check white", "check black", "nothing"]
    data = []
    for i, state in enumerate(states):
        for game_fen in fens[state]:
            one_hot_target = np.zeros((5, 1))
            one_hot_target[i][0] = 1
            data.append((fen_to_vector(game_fen), one_hot_target))

    np.random.shuffle(data)
    return data

def split_data(data, p):
    # Splits the data between training and testing samples
    # p is the percentage of data used for training

    training_samples_count = round(len(data) * p / 100)
    training_data = data[:training_samples_count]
    testing_data = data[training_samples_count:]

    return training_data, testing_data

if __name__ == "__main__":
    # Load the database
    print("Loading data")
    fens = load_data("Positions/positions.json")
    print("Preprocessing")
    data = preprocess_data(fens)
    train_data, test_data = split_data(data, 90)

    """
    # Shit code to find a potential best to train
    for i in range(100):
        hidden_layers = random.randint(1, 10)
        layers = [random.randint(10, 100) for _ in range(hidden_layers)]
        layers.insert(0, 769)
        layers.append(5)
        print(f"Network {i}:", layers)
        net = NeuralNetwork(layers)
        net.AdamWTrain(train_data[:30000], random.randint(16, 128), epochs=2)
        NetworkManager.save_network(net, f"SavedModels/Chess/{i}")
        print("Done.")
        bench = open("benchmark.txt", "a")
        bench.write(f"{net.evaluate(test_data)*100:.2f}\n")
        bench.close()
    """

    net = NetworkManager.load_network("SavedModels/Chess/best_so_far.mnet")
    print(f"{net.evaluate(test_data)*100:.2f}%")

