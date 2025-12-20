import numpy as np
from PIL import Image

MODEL_SIZE = 64

def preprocess_pil_image(pil_img):
    bbox = pil_img.getbbox()
    
    if bbox is None:
        return None
    
    pil_img = pil_img.crop(bbox)
    pil_img = pil_img.resize(
        (MODEL_SIZE, MODEL_SIZE),
        Image.Resampling.LANCZOS
    )
    
    img = np.array(pil_img, dtype = np.float32)
    
    # Invert colors
    img = 255.0 - img
    
    img = img / 255.0
    
    return np.expand_dims(img, axis = 0)

def preprocess_strokes(strokes):
    sequence = []
    
    prev_x, prev_y = 0.0, 0.0
    
    for i, stroke in enumerate(strokes):
        for j, (x, y) in enumerate(stroke):
            dx = x - prev_x
            dy = y - prev_y
            
            pen_down = 1 if j > 0 else 0 
            pen_up = 1 if j == 0 and i > 0 else 0 
            
            if i == 0 and j == 0:
                pen_down, pen_up = 0, 0
            
            sequence.append([dx, dy, pen_down, pen_up])
            prev_x, prev_y = x, y

    dataset = np.array(sequence, dtype=np.float32)

    return dataset