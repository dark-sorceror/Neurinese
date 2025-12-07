import torch
import numpy as np
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageDraw
from tkinter import Canvas, Button

from model import CharacterRecognizer
from preprocess import preprocess_pil_image

CANVAS_SIZE = 300
CHARACTER_TO_COLLECT = "Unknown"
INDEX_OF_CHARACTER = 0
MODEL_SIZE = 64

CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.35

INDEX_TO_CHAR = {
    0: "Unknown",
    1: "你",
    2: "不",
    3: "大"
}

IMAGE_FILE_PATH = Path("./data/image.npy")
LABEL_FILE_PATH = Path("./data/label.npy")
MODEL_PATH = Path("./CNN_char_model.pth")

class DrawingApp:
    def __init__(self, master):
        self.master = master
        
        master.title("Neurinese")

        self.lastX, self.lastY = None, None
        
        self.canvas = Canvas(
            master, 
            width = CANVAS_SIZE, 
            height = CANVAS_SIZE, 
            bg = "black"
        )
        self.canvas.pack(pady = 10)
        
        self.image = Image.new(
            mode = "L", 
            size = (CANVAS_SIZE, CANVAS_SIZE), 
            color = 0
        )
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        
        self.clear_btn = Button(
            master, 
            text = "Clear", 
            command = self.clear_canvas
        )
        self.clear_btn.pack(side = tk.LEFT, padx = 5)
        
        self.save_btn = Button(
            master, 
            text = "Save", 
            command = self.save
        )
        self.save_btn.pack(side = tk.LEFT, padx = 5)
        
        self.recognize_btn = Button(
            master, 
            text = "Recognize", 
            command = self.recognize_char
        )
        self.recognize_btn.pack(side = tk.LEFT, padx = 5)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if MODEL_PATH.exists():
            self.model = CharacterRecognizer(num_classes = len(INDEX_TO_CHAR))
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

    def start_draw(self, event):
        self.lastX, self.lastY = event.x, event.y

    def draw_line(self, event):
        if self.lastX and self.lastY:
            self.canvas.create_line(
                self.lastX, 
                self.lastY, 
                event.x, 
                event.y, 
                fill = "white", 
                width = 7.5, 
                capstyle = tk.ROUND, 
                smooth = tk.TRUE
            )
            self.draw.line(
                [
                    self.lastX, 
                    self.lastY, 
                    event.x, 
                    event.y
                ],
                fill = 255, 
                width = 15
            )
            self.lastX, self.lastY = event.x, event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new(
            mode = "L", 
            size = (CANVAS_SIZE, CANVAS_SIZE), 
            color = 0
        )
        self.draw = ImageDraw.Draw(self.image)
        
    def preprocess_image(self):
        return preprocess_pil_image(self.image)
        
    def save(self):
        processed_input = self.preprocess_image()
        
        if processed_input is None:
            return

        label = np.array(
            [INDEX_OF_CHARACTER], 
            dtype = np.int64
        )
        
        if not IMAGE_FILE_PATH.exists() and not LABEL_FILE_PATH.exists():
            np.save(IMAGE_FILE_PATH, np.expand_dims(processed_input, axis = 0))
            np.save(LABEL_FILE_PATH, label)
        
            print(f"1 total sample")
            
            self.clear_canvas()
            
            return
        
        for i in range(20):
            images = np.load(IMAGE_FILE_PATH)
            labels = np.load(LABEL_FILE_PATH)
            
            image_batch = np.expand_dims(processed_input, axis = 0) 
            
            updated_images = np.concatenate([images, image_batch], axis = 0)
            updated_labels = np.concatenate([labels, label], axis = 0)

            np.save(IMAGE_FILE_PATH, updated_images)
            np.save(LABEL_FILE_PATH, updated_labels)
        
        print(f"{len(images)} total samples")
        
        self.clear_canvas()
        
    def recognize_char(self):
        if not self.model:
            return
        
        input_data = self.preprocess_image()

        input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(self.device)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        probs = torch.softmax(outputs, dim = 1).squeeze().cpu().numpy()
        
        top_idx = probs.argmax()
        top_prob = probs[top_idx]

        sorted_probs = np.sort(probs)
        margin = sorted_probs[-1] - sorted_probs[-2]
        
        if top_idx == 0 or top_prob < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            print(F"Prediction: Unknown Confidence: {top_prob:.2f} Margin: {margin:.2f}")
            
            return

        predicted_char = INDEX_TO_CHAR[top_idx]
        
        print(f"Prediction: {predicted_char} Confidence: {top_prob:.2f} Margin: {margin:.2f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()