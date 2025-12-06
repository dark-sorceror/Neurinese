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
    
    img = np.array(pil_img, dtype=np.float32)
    
    # Invert colors
    img = 255.0 - img
    
    img = img / 255.0
    
    return np.expand_dims(img, axis=0)