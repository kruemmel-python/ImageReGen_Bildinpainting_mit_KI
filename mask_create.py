import os
from PIL import Image
import numpy as np

# Pfad zum Ordner mit den Originalbildern
input_folder = 'D:\\repos\\inpainting\\inpainting\\image_original'

# Pfad zum Ordner, in dem die Masken gespeichert werden sollen
output_folder = 'D:\\repos\\inpainting\\inpainting\\mask_images'

# Erstelle den Ausgabeordner, falls er noch nicht existiert
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Funktion zum Erstellen einer zufälligen Maske für ein Bild
def create_random_mask(image_path, mask_path):
    """
    Erstellt eine zufällige Maske für das gegebene Bild und speichert sie im angegebenen Pfad.
    
    Parameters:
    image_path (str): Pfad zum Originalbild.
    mask_path (str): Pfad, unter dem die Maske gespeichert werden soll.
    """
    # Bild laden und auf feste Größe ohne proportionale Anpassung skalieren
    image = Image.open(image_path).resize((512, 512))
    
    # In ein NumPy-Array konvertieren
    image_array = np.array(image)
    
    # Eine zufällige Maske erstellen
    mask = np.random.choice([0, 255], size=(512, 512), p=[0.75, 0.25]).astype(np.uint8)
    
    # Maske als Bild speichern
    mask_image = Image.fromarray(mask)
    mask_image.save(mask_path)

# Gehe durch alle Bilder im Eingabeordner und erstelle zufällige Masken
for image_file in os.listdir(input_folder):
    # Vollständiger Pfad zum Bild
    image_path = os.path.join(input_folder, image_file)
    
    # Pfad, unter dem die Maske gespeichert werden soll
    mask_path = os.path.join(output_folder, image_file)  # Der Name der Maske entspricht dem Originalbild
    
    # Zufällige Maske für das Bild erstellen
    create_random_mask(image_path, mask_path)

print(f'Zufällige Masken wurden im Ordner "{output_folder}" erstellt.')
