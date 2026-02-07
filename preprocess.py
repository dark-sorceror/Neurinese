import numpy as np
from PIL import Image
from deep_translator import GoogleTranslator

MODEL_SIZE = 64

def preprocess_pil_image(pil_img: Image):
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

def simplify_stroke(stroke, epsilon = 2.0):
    # Ramer-Douglas-Peucker algorithm
    # Reducing the number of points while preserving the shape

    dmax = 0
    index = 0
    end = len(stroke) - 1
    
    p1 = np.array(stroke[0][: 2])
    p2 = np.array(stroke[end][: 2])
    
    if np.array_equal(p1, p2):
        for i in range(1, end):
            d = np.linalg.norm(np.array(stroke[i][: 2]) - p1)
            
            if d > dmax:
                index = i
                dmax = d
    else:
        for i in range(1, end):
            p = np.array(stroke[i][: 2])
            d = np.linalg.norm(np.cross(p2 - p1, p1 - p)) / np.linalg.norm(p2 - p1)
            
            if d > dmax:
                index = i
                dmax = d

    # Recursively simplify
    if dmax > epsilon:
        rec_results1 = simplify_stroke(stroke[: index + 1], epsilon)
        rec_results2 = simplify_stroke(stroke[index :], epsilon)

        return rec_results1[:-1] + rec_results2
    else:
        return [stroke[0], stroke[end]]

def to_relative(seq: np.array):
    out = []
    
    for i in range(1, len(seq)):
        x0, y0, p0 = seq[i - 1]
        x1, y1, p1 = seq[i]
        dx, dy = x1 - x0, y1 - y0
        out.append([dx, dy, p1])
        
    return np.array(out, dtype=np.float32)

def normalize(seq: np.array):
    # Ignore massive movements
    # Means are already relatively small from small movement
    # seq[:, :2] -= seq[:, :2].mean(axis = 0, keepdims = True)

    moves = np.linalg.norm(seq[:, :2], axis=1)
    
    drawing_moves = seq[moves < 50.0] 
    
    if len(drawing_moves) > 0:
        scale = drawing_moves[:, :2].std() + 1e-6
    else:
        scale = seq[:, :2].std() + 1e-6
    
    seq[:, :2] /= scale
    
    return seq

def translate(phrase: str, en: bool):
    return GoogleTranslator(
        source = 'auto',
        target = 'en' if en else 'zh-CN'
    ).translate(phrase)
