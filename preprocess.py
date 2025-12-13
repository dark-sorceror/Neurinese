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
    normalization_factor = 300 / 2.0 # Canvas Size
    
    prev_x, prev_y = 0.0, 0.0
    
    for i, stroke in enumerate(strokes):
        for j, (x, y) in enumerate(stroke):
            dx = (x - prev_x) / normalization_factor
            dy = (y - prev_y) / normalization_factor
            
            pen_start = 1 if (j > 0 or i == 0) else 0
            pen_end = 0
            
            if j == len(stroke) - 1:
                pen_start = 0
                pen_end = 1
            
            sequence.append([dx, dy, pen_start, pen_end])
            prev_x, prev_y = x, y
            
    return np.array(sequence)