import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
from skimage.transform import resize

# Konstanten und Konfigurationen in einem zentralen Dictionary
CONFIG = {
    "model_path": "model-06.keras",  # Pfad zum Modell
    "image_size": (512, 512),        # Größe der Bilder für das Modell
    "min_region_size": 1             # Mindestgröße der Region für Inpainting
}

# Hilfsfunktion zum Laden des Modells
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        messagebox.showerror("Modellfehler", f"Das Modell konnte nicht geladen werden: {e}")
        return None

# Hilfsfunktion zum Erstellen des Menüs
def create_menu(root, callbacks):
    menubar = tk.Menu(root)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="Open", command=callbacks['open_image'])
    menubar.add_cascade(label="File", menu=filemenu)
    root.config(menu=menubar)

# Hauptklasse der Anwendung
class InpaintingApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Inpainting App")
        
        # Lade das Modell
        self.model = load_model(model_path)
        self.image_path = None
        self.image = None
        self.tk_image = None
        
        # Erstelle das Canvas für die Bildanzeige
        self.canvas = tk.Canvas(root, cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Initialisiere Variablen für die Rechteckauswahl
        self.rect = None
        self.start_x = None
        self.start_y = None
        
        # Binde Ereignisse an das Canvas
        self.bind_canvas_events()
        
        # Erstelle das Menü
        create_menu(self.root, {'open_image': self.open_image})
    
    # Methode zum Binden von Canvas-Ereignissen
    def bind_canvas_events(self):
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
    
    # Methode zum Öffnen eines Bildes
    def open_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if self.image_path:
            self.load_and_display_image(self.image_path)
    
    # Methode zum Laden und Anzeigen eines Bildes
    def load_and_display_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.display_image(image)
        except Exception as e:
            messagebox.showerror("Bildfehler", f"Das Bild konnte nicht geladen werden: {e}")
    
    # Methode zum Anzeigen eines Bildes auf dem Canvas
    def display_image(self, img):
        self.image = img
        self.img_height, self.img_width, _ = img.shape
        self.canvas.config(width=self.img_width, height=self.img_height)
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
    
    # Methode, die beim Drücken der Maustaste aufgerufen wird
    def on_button_press(self, event):
        self.start_x = event.x
        self.start_y = event.y
        if not self.rect:
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')
    
    # Methode, die beim Ziehen der Maus mit gedrückter Taste aufgerufen wird
    def on_mouse_drag(self, event):
        cur_x, cur_y = (event.x, event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, cur_x, cur_y)
    
    # Methode, die beim Loslassen der Maustaste aufgerufen wird
    def on_button_release(self, event):
        end_x, end_y = (event.x, event.y)
        self.fill_region(self.start_x, self.start_y, end_x, end_y)
    
    # Methode zum Füllen einer Region mit Inpainting
    def fill_region(self, start_x, start_y, end_x, end_y):
        x1, x2 = sorted([start_x, end_x])
        y1, y2 = sorted([start_y, end_y])
        
        # Stelle sicher, dass die Koordinaten innerhalb der Bildgrenzen liegen
        x1 = max(0, min(x1, self.img_width))
        x2 = max(0, min(x2, self.img_width))
        y1 = max(0, min(y1, self.img_height))
        y2 = max(0, min(y2, self.img_height))
        
        # Stelle sicher, dass die Region eine Mindestgröße hat
        if x2 - x1 < CONFIG['min_region_size'] or y2 - y1 < CONFIG['min_region_size']:
            messagebox.showinfo("Region zu klein", "Die ausgewählte Region ist zu klein.")
            return
        
        region_to_fill = self.image[y1:y2, x1:x2]
        
        # Stelle sicher, dass die Region nicht leer ist
        if region_to_fill.size == 0:
            messagebox.showinfo("Leere Region", "Die Region zum Füllen ist leer.")
            return
        
        # Erstelle eine Maske für die Region
        mask = np.ones(region_to_fill.shape[:2], dtype=np.uint8)
        
        # Wende Inpainting nur auf die Region an
        inpainted_region = self.apply_inpainting(region_to_fill, mask)
        
        # Ersetze die Region im Originalbild durch die inpainted Region
        self.image[y1:y2, x1:x2] = inpainted_region
        self.display_image(self.image)
    
    # Methode zum Anwenden von Inpainting
    def apply_inpainting(self, image, mask):
        # Skaliere Bild und Maske auf die Eingabegröße des Modells
        original_shape = image.shape
        image_resized = resize(image, CONFIG['image_size'], anti_aliasing=True)
        mask_resized = resize(mask, CONFIG['image_size'], anti_aliasing=True, preserve_range=True)
        mask_resized = np.expand_dims(mask_resized, axis=-1)
        
        # Bereite die Eingabe für das Modell vor
        masked_image = image_resized * (1 - mask_resized)  # Wende die Maske auf das Bild an
        masked_image = np.expand_dims(masked_image, axis=0)
        
        # Führe Inpainting durch
        inpainted_image = self.model.predict(masked_image)[0]
        
        # Skaliere das inpainted Bild zurück auf die Originalgröße
        inpainted_image = resize(inpainted_image, original_shape[:2], anti_aliasing=True)
        inpainted_image = (inpainted_image * 255).astype(np.uint8)
        
        return inpainted_image

# Start der Anwendung
if __name__ == "__main__":
    root = tk.Tk()
    app = InpaintingApp(root, CONFIG['model_path'])
    root.mainloop()

