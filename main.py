import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw

CANVAS_SIZE = 300

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
        self.canvas.pack(pady=10)
        
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

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()