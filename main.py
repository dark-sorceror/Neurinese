import time
import torch
import numpy as np
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageDraw
from tkinter import Canvas, Button

from character_model import CharacterRecognizer
from preprocess import preprocess_pil_image, to_relative, normalize, simplify_stroke
from stroke_model import StrokeDataset, StrokeModel
from handwriting_inference import Handwrite

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
STROKE_FILE_PATH = Path("./data/strokes.npy")
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
        
        self.draw_btn = Button(
            master, 
            text = "Draw", 
            command = self.draw_char
        )
        self.draw_btn.pack(side = tk.LEFT, padx = 5)
        
        self.generate_btn = Button(
            master, 
            text = "Generate", 
            command = self.generate_char
        )
        self.generate_btn.pack(side = tk.LEFT, padx = 5)
        
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
    
    def preprocess_strokes(self):
        seq = []
        
        for stroke in self.strokes:
            raw_stroke = []
            
            for x, y in stroke:
                raw_stroke.append([x, y])
            
            simple_stroke = simplify_stroke(raw_stroke, epsilon = 2.0)
            
            for i, (x, y) in enumerate(simple_stroke):
                if i == 0: p = 1
                else: p = 0
                    
                seq.append([x, y, p])
                
        return np.array(seq, dtype = np.float32)
        
    def save(self):
        processed_input = self.preprocess_image()
        
        if processed_input is None: return
        
        image_batch = np.repeat(np.expand_dims(processed_input, axis = 0), 20, axis = 0)
        label_batch = np.full(10, self.INDEX_OF_CHARACTER, dtype = np.int64)
        
        if not IMAGE_FILE_PATH.exists():
            np.save(IMAGE_FILE_PATH, image_batch)
            np.save(LABEL_FILE_PATH, label_batch)
            
            print(f"Saved character drawing. Total samples: {image_batch.shape[0]}")
        else:
            images = np.load(IMAGE_FILE_PATH)
            labels = np.load(LABEL_FILE_PATH)
            
            updated_image_batch = np.concatenate([images, image_batch], axis = 0)
            updated_labels_batch = np.concatenate([labels, label_batch], axis = 0)
            
            np.save(IMAGE_FILE_PATH, updated_image_batch)
            np.save(LABEL_FILE_PATH, updated_labels_batch)
            
            print(f"Saved character drawing. Total samples: {updated_image_batch.shape[0]}")

        seq = self.preprocess_strokes()
        
        if len(seq) == 0: return

        if not STROKE_FILE_PATH.exists():
            stroke_batch = [seq]
        else:
            strokes = np.load(STROKE_FILE_PATH, allow_pickle = True)
            stroke_batch = list(strokes)
            
            for _ in range(20):
                stroke_batch.append(seq)

        stroke_array = np.array(stroke_batch, dtype = object)
        np.save(STROKE_FILE_PATH, stroke_array)
    
        print(f"Saved vector stroke. Total samples: {len(stroke_batch)}")
        
        # No conditional VAE yet; overfitting on a single sample
        """
        if self.INDEX_OF_CHARACTER == len(self.INDEX_TO_CHAR) - 1:
            self.INDEX_OF_CHARACTER = 0
        else:
            self.INDEX_OF_CHARACTER += 1
        """
            
        self.CHARACTER_TO_COLLECT = self.INDEX_TO_CHAR.get(self.INDEX_OF_CHARACTER)
        
        print(f"Next character: {self.CHARACTER_TO_COLLECT}")
        
        self.strokes.clear()
        self.clear_canvas()
    
    @torch.no_grad()    
    def recognize_char(self):
        if not self.model:
            return
        
        input_data = self.preprocess_image()

        input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(self.device)
        input_tensor = input_tensor.to(self.device)
        
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
    
    @torch.no_grad() 
    def draw_char(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = StrokeModel(
            input_size = 3,
            hidden_size = 256,
            latent_size = 64,
            num_layers = 1
        ).to(device)

        model.load_state_dict(torch.load(
                "models/handwriting_model.pth", 
                map_location = device
            )
        )

        generator = Handwrite(model = model, device = device)

        samples = np.load("./data/strokes.npy", allow_pickle = True)
        single_sample = [to_relative(normalize(samples[0].astype(np.float32)))]

        dataset = StrokeDataset(single_sample) 
        sample_tensor = dataset[0] .unsqueeze(0).to(device)

        model.eval()
        mean_dist, log_var = model.encoder(sample_tensor)
        z = mean_dist

        # gen_strokes = generator.generate(z = z)
            
        seq = sample_tensor.cpu().numpy()

        min_x, max_x = 0, 0
        min_y, max_y = 0, 0
        curr_x, curr_y = 0, 0

        for dx, dy, pen in seq:
            curr_x += dx * 0.6
            curr_y += dy * 0.6
            
            min_x, max_x = min(min_x, curr_x), max(max_x, curr_x)
            min_y, max_y = min(min_y, curr_y), max(max_y, curr_y)

        self.canvas.update_idletasks() 
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        drawing_w = max_x - min_x
        drawing_h = max_y - min_y

        start_x = (canvas_w - drawing_w) / 2 - min_x
        start_y = (canvas_h - drawing_h) / 2 - min_y

        x, y = start_x, start_y

        def draw_step(index, curr_x, curr_y):
            if index >= len(seq):
                return

            dx, dy, pen = seq[index]

            nx = curr_x + (dx * 0.6)
            ny = curr_y + (dy * 0.6)

            if pen < 0.5:
                self.canvas.create_line(
                    x0 = curr_x,
                    y0 = curr_y,
                    x1 = nx,
                    y1 = ny,
                    fill = "white",
                    width = 7.5, 
                    capstyle = tk.ROUND, 
                    smooth = tk.TRUE
                )
            
            self.canvas.after(10, draw_step, index + 1, nx, ny)

        draw_step(0, x, y)
    
    @torch.no_grad()
    def generate_char(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = StrokeModel(
            input_size = 3,
            hidden_size = 256,
            latent_size = 64,
            num_layers = 1
        ).to(device)

        model.load_state_dict(torch.load(
                "models/handwriting_model.pth", 
                map_location = device
            )
        )

        generator = Handwrite(model = model, device = device)

        samples = np.load("./data/strokes.npy", allow_pickle = True)
        single_sample = [to_relative(normalize(samples[0].astype(np.float32)))]

        dataset = StrokeDataset(single_sample) 

        sample_tensor = dataset[0].unsqueeze(0).to(device)

        model.eval()
        
        mean_dist, log_var = model.encoder(sample_tensor)
        z = mean_dist

        # gen_strokes = generator.generate(z = z)
            
        seq = generator.reconstruct(sample_tensor)

        min_x, max_x = 0, 0
        min_y, max_y = 0, 0
        curr_x, curr_y = 0, 0

        for dx, dy, pen in seq:
            curr_x += dx * 0.6
            curr_y += dy * 0.6
            
            min_x, max_x = min(min_x, curr_x), max(max_x, curr_x)
            min_y, max_y = min(min_y, curr_y), max(max_y, curr_y)

        self.canvas.update_idletasks() 
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        drawing_w = max_x - min_x
        drawing_h = max_y - min_y

        start_x = (canvas_w - drawing_w) / 2 - min_x
        start_y = (canvas_h - drawing_h) / 2 - min_y

        x, y = start_x, start_y

        def draw_step(index, curr_x, curr_y):
            if index >= len(seq):
                return

            dx, dy, pen = seq[index]

            nx = curr_x + (dx * 0.6)
            ny = curr_y + (dy * 0.6)

            if pen < 0.5:
                self.canvas.create_line(
                    x0 = curr_x,
                    y0 = curr_y,
                    x1 = nx,
                    y1 = ny,
                    fill = "white",
                    width = 7.5, 
                    capstyle = tk.ROUND, 
                    smooth = tk.TRUE
                )
            
            self.canvas.after(10, draw_step, index + 1, nx, ny)

        draw_step(0, x, y)

if __name__ == "__main__":
    print("Character to collect: 你")
    
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()