import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "./data/strokes.npy"

def plot_strokes(seq: list[list[int]]):
    plt.figure(figsize = (6, 6))
    
    for i in seq:
        x, y = 0.0, 0.0

        for dx, dy, pen in i:
            nx, ny = x + dx, y + dy

            if pen > 0.5:
                plt.plot([x, nx], [y, ny], 'k-', linewidth = 2)

            x, y = nx, ny

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.axis('off')
    plt.show()

def normalize(seq: list[list[int]]):
    seq[:, :2] -= seq[:, :2].mean(axis = 0, keepdims = True)
    scale = seq[:, :2].std() + 1e-6
    seq[:, :2] /= scale
    
    return seq

def to_relative(stroke: list[list[int]]):
    out = []
    
    for i in range(1, len(stroke)):
        x0, y0, p0 = stroke[i - 1]
        x1, y1, p1 = stroke[i]
        dx, dy = x1 - x0, y1 - y0
        
        out.append([dx, dy, p1])
        
    return out

if __name__ == "__main__":
    samples = np.load("./data/strokes.npy", allow_pickle=True)
    raw_data = [seq.astype(np.float32) for seq in samples]

    processed_samples = []

    for raw in raw_data:
        seq_abs = normalize(raw)
        seq = to_relative(seq_abs)

        processed_samples.append(seq)
        
    plot_strokes(processed_samples[:120:20])