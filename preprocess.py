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

def to_relative(seq):
    out = []
    
    for i in range(1, len(seq)):
        x0, y0, p0 = seq[i - 1]
        x1, y1, p1 = seq[i]
        dx, dy = (x1 - x0) * 100, (y1 - y0) * 100
        
        out.append([dx, dy, p1])
        
    return np.array(out, dtype = np.float32)

def normalize(seq):
    seq[:, :2] -= seq[:, :2].mean(axis = 0, keepdims = True)
    scale = seq[:, :2].std() + 1e-6
    seq[:, :2] /= scale
    
    return seq