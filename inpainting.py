# Importieren der benötigten Bibliotheken
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import datetime
import logging

# Konfiguration des Logging-Moduls für bessere Fehlerverfolgung und Ausgabe
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Setzen der Umgebungsvariable für oneDNN auf '0', um die Nutzung zu deaktivieren
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Funktion zum Laden und Vorbereiten der Bilder
def load_and_prepare_images(image_folder, mask_folder, image_size):
    """
    Lädt und bereitet die Bilder für das Training vor.
    
    Parameters:
    image_folder (str): Pfad zum Ordner mit den Originalbildern.
    mask_folder (str): Pfad zum Ordner mit den Maskenbildern.
    image_size (tuple): Zielgröße der Bilder (Höhe, Breite, Kanäle).
    
    Returns:
    np.array: Array von vorbereiteten Originalbildern.
    np.array: Array von vorbereiteten Maskenbildern.
    """
    original_images = []
    masked_images = []
    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        mask_path = os.path.join(mask_folder, img_name)
        try:
            # Originalbild laden und skalieren
            img = imread(img_path)
            img = resize(img, image_size, anti_aliasing=True)
            original_images.append(img)

            # Maskenbild laden und skalieren, as_gray explizit auf True setzen
            mask = imread(mask_path, as_gray=True)
            mask = resize(mask, image_size[:2], anti_aliasing=True)

            # Maske auf das Originalbild anwenden
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)  # Sicherstellen, dass die Maske 3 Kanäle hat
            masked_img = img * mask
            masked_images.append(masked_img)
        except IOError as e:
            # Spezifischere Fehlerbehandlung für Ein-/Ausgabeoperationen
            logging.error(f"Fehler beim Laden oder Verarbeiten von {img_path}: {e}")
    
    logging.info(f"Loaded {len(original_images)} images from {image_folder}")
    return np.array(original_images), np.array(masked_images)

# Funktion zum Normalisieren der Bilddaten
def normalize_images(images):
    """
    Normalisiert die Bilddaten auf den Bereich [0, 1].
    
    Parameters:
    images (np.array): Array von Bildern.
    
    Returns:
    np.array: Array von normalisierten Bildern.
    """
    images = images.astype('float32')
    normalized_images = images / 255.0
    return normalized_images

# Hauptteil des Skripts
if __name__ == "__main__":
    # Pfad zum Ordner mit den Originalbildern, relativ zum aktuellen Arbeitsverzeichnis
    image_folder = 'image_original'
    # Pfad zum Ordner mit den Maskenbildern, relativ zum aktuellen Arbeitsverzeichnis
    mask_folder = 'mask'

    # Bilder laden und vorbereiten
    x_data, y_data = load_and_prepare_images(image_folder, mask_folder, image_size=(512, 512, 3))

    # Aufteilen in Trainings- und Validierungsdaten
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    # Normalisiere die Bilddaten
    x_train_normalized = normalize_images(x_train)
    y_train_normalized = normalize_images(y_train)
    x_val_normalized = normalize_images(x_val)
    y_val_normalized = normalize_images(y_val)

    # Eingabeschicht definieren
    input_img = Input(shape=(512, 512, 3))

    # Encoder
    x = Conv2D(64, (3, 3), padding='same')(input_img)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    x = Activation('sigmoid')(x)

    # Modell definieren
    model = Model(input_img, x)

    # Aktuelles Datum und Uhrzeit für den Namen des Log-Verzeichnisses
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + current_time

    # TensorBoard Callback erstellen
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Callbacks definieren
    checkpoint_path = "model-{epoch:02d}.keras"
    checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # Modell kompilieren
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_absolute_error')

    # Modell trainieren und Callbacks hinzufügen
    model.fit(
        x_train_normalized,
        y_train_normalized,
        validation_data=(x_val_normalized, y_val_normalized),
        epochs=100,
        callbacks=[checkpoint, early_stopping, tensorboard_callback, reduce_lr],
        verbose=1
    )
