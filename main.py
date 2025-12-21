import time
import torch
import numpy as np
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageDraw
from tkinter import Canvas, Button

from character_model import CharacterRecognizer
from preprocess import preprocess_pil_image, preprocess_strokes

CANVAS_SIZE = 300
MODEL_SIZE = 64

CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.35

INDEX_TO_CHAR = {
    0: "你",
    1: "不",
    2: "大"
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
        
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)
        
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
        
        self.test_btn = Button(
            master, 
            text = "Draw", 
            command = self.test
        )
        self.test_btn.pack(side = tk.LEFT, padx = 5)
        
        self.CHARACTER_TO_COLLECT = "你"
        self.INDEX_OF_CHARACTER = 0
        self.INDEX_TO_CHAR = {
            0: "你",
            1: "不",
            2: "大"
        }
        
        self.strokes = []
        self.current_stroke = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if MODEL_PATH.exists():
            self.model = CharacterRecognizer(num_classes = len(INDEX_TO_CHAR))
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

    def start_stroke(self, event):
        self.lastX, self.lastY = event.x, event.y
        
        self.canvas.create_oval(
            event.x - 3.75, 
            event.y - 3.75, 
            event.x + 3.75, 
            event.y + 3.75,
            fill = "white", 
            outline = "white" 
        )

        self.current_stroke.append((event.x, event.y))
        
    def end_stroke(self, event):
        print(self.current_stroke)
        
        if self.current_stroke:
            self.strokes.append(self.current_stroke[:])
        
        self.current_stroke.clear()

    def draw_line(self, event):
        self.current_stroke.append((event.x, event.y))
        
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
            [self.INDEX_OF_CHARACTER], 
            dtype = np.int64
        )
        
        if not IMAGE_FILE_PATH.exists() and not LABEL_FILE_PATH.exists():
            np.save(IMAGE_FILE_PATH, np.expand_dims(processed_input, axis = 0))
            np.save(LABEL_FILE_PATH, label)
        
        for i in range(50):
            images = np.load(IMAGE_FILE_PATH)
            labels = np.load(LABEL_FILE_PATH)
            
            image_batch = np.expand_dims(processed_input, axis = 0) 
            
            updated_images = np.concatenate([images, image_batch], axis = 0)
            updated_labels = np.concatenate([labels, label], axis = 0)

            np.save(IMAGE_FILE_PATH, updated_images)
            np.save(LABEL_FILE_PATH, updated_labels)
        
        print(f"{len(images)} total samples")
        
        strokes = preprocess_strokes(self.strokes)
        
        #print(strokes)

        if self.INDEX_OF_CHARACTER == len(self.INDEX_TO_CHAR) - 1:
            self.INDEX_OF_CHARACTER = 0
        else:
            self.INDEX_OF_CHARACTER += 1
        
        self.CHARACTER_TO_COLLECT = self.INDEX_TO_CHAR.get(self.INDEX_OF_CHARACTER)
        
        print(f"Character to collect: {self.CHARACTER_TO_COLLECT}")
        
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
        
        if top_prob < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            print(F"Prediction: Unknown\tConfidence: {top_prob:.2f}\tMargin: {margin:.2f}")
            
            return

        predicted_char = self.INDEX_TO_CHAR[top_idx]
        
        print(f"Prediction: {predicted_char}\tConfidence: {top_prob:.2f}\tMargin: {margin:.2f}")
        
    def test(self):
        self.canvas.delete("all")
        
        self.lastX = 0
        self.lastY = 0
        
        strokes = preprocess_strokes(self.strokes).tolist()
        
        print(strokes)

        # Center of mass lol
        abs_coors = []
        c_x, c_y = 0, 0

        for stroke in strokes:
            c_x += stroke[0] * 1
            c_y += stroke[1] * 1
            
            abs_coors.append((c_x, c_y))
        
        min_x, max_x = min([p[0] for p in abs_coors]), max([p[0] for p in abs_coors])
        min_y, max_y = min([p[1] for p in abs_coors]), max([p[1] for p in abs_coors])

        drawing_width = max_x - min_x
        drawing_height = max_y - min_y

        offset_x = 150 - (min_x + drawing_width / 2)
        offset_y = 150 - (min_y + drawing_height / 2)

        self.lastX = 0 + offset_x
        self.lastY = 0 + offset_y
        
        for stroke in strokes: 
            x = self.lastX + stroke[0]
            y = self.lastY + stroke[1]
            
            if stroke[2]:
                self.canvas.create_line(
                    self.lastX, 
                    self.lastY, 
                    x,
                    y,
                    fill = "white", 
                    width = 7.5,
                    capstyle = tk.ROUND, 
                    smooth = tk.TRUE
                )
                
            self.lastX = x
            self.lastY = y

if __name__ == "__main__":
    print("Character to collect: 你")
    
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()