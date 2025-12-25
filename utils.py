import matplotlib.pyplot as plt

def plot_strokes(seq: list[list[list[int]]], multiple: bool = True):
    plt.figure(figsize = (6, 6))
    
    if not multiple:
        seq = [seq]
    
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