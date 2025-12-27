import numpy as np
from utils import plot_strokes
from preprocess import normalize, to_relative

DATA_PATH = "./data/strokes.npy"

if __name__ == "__main__":
    samples = np.load(DATA_PATH, allow_pickle = True)
    raw_data = [seq.astype(np.float32) for seq in samples]
    
    processed_samples = []

    for raw in raw_data:
        seq_abs = normalize(raw)
        seq = to_relative(seq_abs)

        processed_samples.append(seq)
        
    plot_strokes(processed_samples[0:2])