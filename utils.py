import matplotlib.pyplot as plt

def plot_strokes(seq: list[list[list[int]]], multiple: bool = True):
    plt.figure(figsize = (6, 6))
    
    if not multiple:
        seq = [seq]
        
    for i in seq:
        x, y = 0.0, 0.0
        
        for step in i:
            dx, dy, p1, p2, p3 = step
            nx, ny = x + dx, y + dy
            
            if p1 == 1:
                plt.plot([x, nx], [y, ny], 'k-', linewidth = 2)
            elif p2 == 1:
                plt.plot([x, nx], [y, ny], color = '0.9', linewidth = 2)
            
            x, y = nx, ny
            
            if p3 == 1:
                break

    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.show()