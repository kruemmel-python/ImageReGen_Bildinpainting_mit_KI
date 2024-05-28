from PIL import Image
import os

# Definieren Sie den Pfad zum Quellordner und zum Zielordner
source_folder = 'D:\\repos\\inpainting\\inpainting\\mask_images'
target_folder = 'D:\\repos\inpainting\\inpainting\\mask'
target_format = 'png'  # Zielbildformat, z.B. 'JPEG', 'PNG', etc.
target_size = (512, 512)  # Zielgröße als Tuple (Breite, Höhe)
                                                                
# Erstellen Sie den Zielordner, falls er nicht existiert
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Durchlaufen Sie alle Dateien im Quellordner
for file_name in os.listdir(source_folder):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Öffnen Sie das Bild
        with Image.open(os.path.join(source_folder, file_name)) as img:
            # Konvertieren Sie das Bild ohne Proportionalität zu berücksichtigen
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            # Speichern Sie das Bild im neuen Format
            img_resized.save(os.path.join(target_folder, f'{os.path.splitext(file_name)[0]}.{target_format.lower()}'), target_format)

print('Bildkonvertierung abgeschlossen.')
