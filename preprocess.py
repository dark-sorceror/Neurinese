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

def to_relative_strokes(strokes: list[list[int]]):
    sequence = []
    
    prev_x, prev_y = 0.0, 0.0

    for x, y, p in strokes:
        dx, dy = x - prev_x, y - prev_y
        sequence.append([dx, dy, p])
        prev_x, prev_y = x, y

    dataset = np.array(sequence, dtype = np.float32)

    return dataset