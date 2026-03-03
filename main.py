import time
import torch
import numpy as np
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageDraw
from tkinter import Canvas, Button

from character_model import CharacterRecognizer
from preprocess import preprocess_pil_image, to_relative, normalize
from stroke_model import StrokeDataset, StrokeModel
from handwriting_inference import Handwrite

CANVAS_SIZE = 300
MODEL_SIZE = 64

CONFIDENCE_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.35

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
        
        self.complete_btn = Button(
            master, 
            text = "Use Style", 
            command = self.infer_style
        )
        self.complete_btn.pack(side = tk.LEFT, padx = 5)
        
        self.CHARACTER_TO_COLLECT = "我"
        self.INDEX_OF_CHARACTER = 2
        self.INDEX_TO_CHAR = {
            0: "你",
            1: "大",
            2: "我"
        }
        
        self.strokes = []
        self.current_stroke = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if MODEL_PATH.exists():
            self.model = CharacterRecognizer(num_classes = len(self.INDEX_TO_CHAR))
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
                raw_stroke.append([float(x), float(y)])
            
            if len(raw_stroke) > 0:
                seq.append(raw_stroke)
                
        return seq
        
    def save(self):
        processed_input = self.preprocess_image()
        
        if processed_input is None: return

        raw_seq = self.preprocess_strokes()
        
        if len(raw_seq) == 0: return

        seq = to_relative(raw_seq).tolist()

        new_images = [processed_input]
        
        for _ in range(19):
            scale = np.random.uniform(0.85, 1.15)
            noise = np.random.normal(0, 5, size=processed_input.shape)
            aug_img = np.clip(processed_input * scale + noise, 0, 1)
            new_images.append(aug_img)

        new_images = np.array(new_images)
        new_labels = np.full(len(new_images), self.INDEX_OF_CHARACTER)

        if not IMAGE_FILE_PATH.exists():
            np.save(IMAGE_FILE_PATH, new_images)
            np.save(LABEL_FILE_PATH, new_labels)
        else:
            images = np.load(IMAGE_FILE_PATH)
            labels = np.load(LABEL_FILE_PATH)
            
            np.save(IMAGE_FILE_PATH, np.concatenate([images, new_images], axis = 0))
            np.save(LABEL_FILE_PATH, np.concatenate([labels, new_labels], axis = 0))

        def augment_stroke(seq: list):
            scale = np.random.uniform(0.85, 1.15)
            angle = np.random.uniform(-0.05, 0.05)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            aug = []
            
            for step in seq:
                dx, dy = float(step[0]), float(step[1])
                p1, p2, p3 = float(step[2]), float(step[3]), float(step[4])
                new_dx = (cos_a * dx - sin_a * dy) * scale + np.random.normal(0, 0.03)
                new_dy = (sin_a * dx + cos_a * dy) * scale + np.random.normal(0, 0.03)
                aug.append([new_dx, new_dy, p1, p2, p3])
                
            return aug

        if STROKE_FILE_PATH.exists():
            stroke_batch = list(np.load(STROKE_FILE_PATH, allow_pickle = True))
        else:
            stroke_batch = []

        stroke_batch.append((seq, self.INDEX_OF_CHARACTER))
        
        for _ in range(19):
            stroke_batch.append((augment_stroke(seq), self.INDEX_OF_CHARACTER))

        np.save(STROKE_FILE_PATH, np.array(stroke_batch, dtype = object))
        
        print(f"Saved 1 + 19 augmented. Total samples: {len(stroke_batch)}")

        self.strokes.clear()
        self.clear_canvas()
    
    def animate(self, seq):
        self.clear_canvas()

        curr_x, curr_y = 0.0, 0.0
        min_x, max_x, min_y, max_y = 0.0, 0.0, 0.0, 0.0

        for step in seq:
            dx, dy = step[0], step[1]
            curr_x += dx * 50
            curr_y += dy * 50
            min_x, max_x = min(min_x, curr_x), max(max_x, curr_x)
            min_y, max_y = min(min_y, curr_y), max(max_y, curr_y)

        self.canvas.update_idletasks()
        
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        start_x = (canvas_w - (max_x - min_x)) / 2 - min_x
        start_y = (canvas_h - (max_y - min_y)) / 2 - min_y

        def draw_step(index, curr_x, curr_y):
            if index >= len(seq):
                return

            step = seq[index]
            dx, dy = step[0], step[1]
            p1 = step[2]

            nx = curr_x + dx * 50
            ny = curr_y + dy * 50

            if p1 > 0.5:
                self.canvas.create_line(
                    curr_x,
                    curr_y, 
                    nx, 
                    ny,
                    fill = "white",
                    width = 7.5,
                    capstyle = tk.ROUND, 
                    smooth = tk.TRUE
                )

            self.canvas.after(10, draw_step, index + 1, nx, ny)

        draw_step(0, start_x, start_y)
    
    def infer_style(self):
        seq = self.preprocess_strokes()
        
        if not seq: return

        relative = to_relative(seq)
        normalized = normalize(relative)
        single_sample = [(normalized, 0)]

        dataset = StrokeDataset(single_sample)
        sample_tensor, char_id_tensor = dataset[0]

        model = StrokeModel(
            input_size = 5, 
            hidden_size = 256, 
            latent_size = 64,
            num_layers = 1, 
            num_classes = 10, 
            class_emb_dim = 32
        ).to(self.device)
        model.load_state_dict(
            torch.load("models/handwriting_model.pth", 
                map_location = self.device
            )
        )

        generator = Handwrite(model = model, device = self.device)

        with torch.no_grad():
            seq_t = sample_tensor.unsqueeze(0).to(self.device)
            cid_t = char_id_tensor.unsqueeze(0).to(self.device)
            mu, _ = model.encoder(seq_t, cid_t)
            
            print(f"z_style mean magnitude: {mu.abs().mean().item():.4f}")
            print(f"z_style std: {mu.std().item():.4f}")
            
        ac = generator.autocomplete(sample_tensor, char_id_tensor, next_char_id = 2)
        
        self.animate(ac)
    
    @torch.no_grad()    
    def recognize_char(self):
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
    def generate_char(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = StrokeModel(
            input_size = 5,
            hidden_size = 256,
            latent_size = 64,
            num_layers = 1,
            num_classes = 10,
            class_emb_dim = 32
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