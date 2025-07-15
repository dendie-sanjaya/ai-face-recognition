# convert.py - Converts Photos to Face Vectors and Stores them in Annoy Index
# This file can be triggered by watcher.py or run independently for bulk conversion.

import os
import sys
import numpy as np
import json
from annoy import AnnoyIndex
import sqlite3
import cv2 # Import OpenCV for image processing
from mtcnn.mtcnn import MTCNN # For face detection
from keras_facenet import FaceNet # For FaceNet model

# --- Configuration ---
PHOTO_STORAGE_FOLDER = 'data_foto'
ANNOY_INDEX_PATH = 'database_foto_vector/face_vectors.ann'
ANNOY_ID_MAP_PATH = 'database_foto_vector/face_id_map.json'

# SQLite Database for User Profiles
DATABASE_USER_PROFILE = 'database_user/user_profiles.db'

# Annoy Index configuration
VECTOR_DIMENSION = 512 # <<< PENTING: Ganti ke 512 untuk FaceNet terbaru
METRIC = 'angular'
N_TREES = 10

# Supported image file extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

# --- Inisialisasi Model FaceNet dan MTCNN secara Global ---
# Model ini akan dimuat sekali saat program dimulai
print("INFO: Memuat model FaceNet dan MTCNN...")
# Jika ada masalah memori atau load_model, bisa coba pakai FaceNet() tanpa argumen.
# Atau pastikan TensorFlow sudah diinstal dengan benar.
try:
    facenet_model = FaceNet()
    mtcnn_detector = MTCNN()
    print("INFO: Model FaceNet dan MTCNN berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model FaceNet atau MTCNN: {e}")
    print("Pastikan TensorFlow dan pustaka terkait sudah terinstal dengan benar.")
    sys.exit(1) # Keluar dari program jika model tidak bisa dimuat

# --- SQLite Database Functions ---
def get_user_profile_db_connection():
    """Establishes a connection to the user profiles SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_USER_PROFILE)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"ERROR: Could not connect to database at {DATABASE_USER_PROFILE}: {e}")
        return None

def update_user_face_id(user_id, annoy_id):
    """
    Updates the 'face_id' column for a given 'user_id' in the 'users' table.
    If the user_id does not exist, it will attempt to insert a new record with default values.
    """
    conn = get_user_profile_db_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET face_id = ? WHERE id = ?", (annoy_id, user_id))
        conn.commit()
        
        if cursor.rowcount > 0:
            print(f"INFO: Successfully updated face_id for existing user '{user_id}' to Annoy ID '{annoy_id}'.")
            return True
        else:
            print(f"INFO: User ID '{user_id}' not found in database. Attempting to insert a new user record.")
            try:
                cursor.execute(
                    "INSERT INTO users (id, name, email, face_id) VALUES (?, ?, ?, ?)",
                    (user_id, f"User {user_id}", f"{user_id}@example.com", annoy_id)
                )
                conn.commit()
                print(f"INFO: Successfully inserted new user '{user_id}' with Annoy ID '{annoy_id}'.")
                return True
            except sqlite3.IntegrityError:
                print(f"WARNING: User ID '{user_id}' already exists but could not be updated or re-inserted. Skipping.")
                return False
    except sqlite3.Error as e:
        print(f"ERROR: Database operation failed for user '{user_id}': {e}")
        return False
    finally:
        if conn:
            conn.close()

# --- Fungsi Ekstraksi Embedding Wajah Menggunakan FaceNet (UPDATED) ---
def extract_face_embedding(image_path):
    """
    Ekstraksi embedding wajah dari sebuah gambar menggunakan MTCNN untuk deteksi
    dan FaceNet untuk embedding.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"ERROR: Gagal membaca gambar: {image_path}")
            return None

        # Deteksi wajah menggunakan MTCNN
        faces = mtcnn_detector.detect_faces(img)

        if not faces:
            print(f"WARNING: Tidak ada wajah terdeteksi di {image_path}.")
            return None
        
        # Asumsi hanya ada satu wajah utama per foto profil.
        # Atau Anda bisa memilih wajah dengan area terbesar jika ada beberapa.
        x, y, width, height = faces[0]['box']
        
        # Ekstrak wajah dari gambar
        # Pastikan koordinat valid
        x1, y1 = abs(x), abs(y)
        x2, y2 = abs(x) + width, abs(y) + height
        face_img = img[y1:y2, x1:x2]

        # Ubah ukuran wajah ke ukuran input yang dibutuhkan FaceNet (160x160)
        face_img = cv2.resize(face_img, (160, 160))
        
        # Ekstraksi embedding menggunakan FaceNet
        # Method 'embeddings' dari keras-facenet mengharapkan list of arrays
        embedding = facenet_model.embeddings([face_img])[0]
        
        return embedding.tolist() # Kembalikan sebagai list

    except Exception as e:
        print(f"ERROR: Gagal mengekstrak embedding dari '{image_path}': {e}")
        return None

# --- Annoy Index & Map Management Functions ---
def load_id_map():
    """Loads the Annoy ID map, or initializes it if it doesn't exist."""
    id_map = {}
    user_id_to_annoy_id = {}

    if os.path.exists(ANNOY_ID_MAP_PATH):
        try:
            with open(ANNOY_ID_MAP_PATH, 'r') as f:
                id_map = json.load(f)
                user_id_to_annoy_id = {v: int(k) for k, v in id_map.items()}
            print(f"INFO: ID map loaded from {ANNOY_ID_MAP_PATH}")
        except Exception as e:
            print(f"WARNING: Error loading ID map from {ANNOY_ID_MAP_PATH}: {e}. Initializing an empty map.")
            id_map = {}
            user_id_to_annoy_id = {}
    else:
        print("INFO: ID map not found, initializing empty map.")
    return id_map, user_id_to_annoy_id

def save_annoy_index_and_map(annoy_index, id_map):
    """Saves the Annoy index and the ID map to disk."""
    try:
        if annoy_index.get_n_items() > 0:
            annoy_index.build(N_TREES)
            annoy_index.save(ANNOY_INDEX_PATH)
            print(f"INFO: Annoy index saved to {ANNOY_INDEX_PATH}")
        else:
            print("INFO: Annoy index is empty, skipping build and save.")

        with open(ANNOY_ID_MAP_PATH, 'w') as f:
            json.dump(id_map, f, indent=4)
        print(f"INFO: ID map saved to {ANNOY_ID_MAP_PATH}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save Annoy index or ID map: {e}")
        return False

# --- Main Conversion Logic ---
def convert_and_store_photo(image_filename):
    """
    Takes an image filename, extracts its face vector using FaceNet,
    stores it in the Annoy index, and updates the associated user's face_id in SQLite.
    """
    user_id = os.path.splitext(image_filename)[0]
    image_path = os.path.join(PHOTO_STORAGE_FOLDER, image_filename)

    if not os.path.exists(image_path):
        print(f"ERROR: Photo file not found at {image_path}. Skipping conversion.")
        return False

    try:
        face_vector = extract_face_embedding(image_path)
        if face_vector is None:
            print(f"WARNING: No valid face embedding extracted for '{image_filename}'. Skipping.")
            return False
        
        id_map, user_id_to_annoy_id = load_id_map()
        temp_annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)

        if os.path.exists(ANNOY_INDEX_PATH):
            old_annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
            try:
                old_annoy_index.load(ANNOY_INDEX_PATH)
                for annoy_id_str, existing_user_id in id_map.items():
                    existing_annoy_id = int(annoy_id_str)
                    if existing_user_id != user_id:
                        temp_annoy_index.add_item(existing_annoy_id, old_annoy_index.get_item_vector(existing_annoy_id))
                    else:
                        print(f"INFO: Skipping old vector for user '{user_id}' as it will be updated with a new one.")
            except Exception as e:
                print(f"WARNING: Could not load existing Annoy index for rebuilding ({e}). Starting with an empty index.")
                id_map = {}
                user_id_to_annoy_id = {}

        annoy_id = user_id_to_annoy_id.get(user_id)
        is_new_user_annoy_entry = False
        if annoy_id is None:
            annoy_id = temp_annoy_index.get_n_items()
            is_new_user_annoy_entry = True
            
        temp_annoy_index.add_item(annoy_id, face_vector)
        
        id_map[str(annoy_id)] = user_id
        user_id_to_annoy_id[user_id] = annoy_id

        if is_new_user_annoy_entry:
            print(f"INFO: New Annoy vector added for user_id '{user_id}' with Annoy ID '{annoy_id}'.")
        else:
            print(f"INFO: Annoy vector updated for user_id '{user_id}' with Annoy ID '{annoy_id}'.")

        if update_user_face_id(user_id, annoy_id):
            print(f"INFO: Successfully linked Annoy ID '{annoy_id}' to user '{user_id}' in SQLite database.")
        else:
            print(f"ERROR: Failed to link Annoy ID '{annoy_id}' to user '{user_id}' in SQLite database.")

        return save_annoy_index_and_map(temp_annoy_index, id_map)

    except Exception as e:
        print(f"FATAL ERROR: Processing photo '{image_filename}' failed: {e}")
        return False

# --- Main Execution Block ---
if __name__ == '__main__':
    conn = None
    try:
        os.makedirs(os.path.dirname(DATABASE_USER_PROFILE), exist_ok=True)
        conn = sqlite3.connect(DATABASE_USER_PROFILE)
        print("INFO: Confirmed connection to 'user_profiles.db'. Assuming 'users' table with 'face_id' column exists.")
    except sqlite3.Error as e:
        print(f"CRITICAL ERROR: SQLite database connection failed: {e}")
    finally:
        if conn:
            conn.close()

    os.makedirs(os.path.dirname(ANNOY_INDEX_PATH), exist_ok=True)

    if len(sys.argv) == 2:
        filename_to_convert = sys.argv[1]
        print(f"\n--- Starting conversion for single file: '{filename_to_convert}' ---")
        convert_and_store_photo(filename_to_convert)
        print(f"--- Finished conversion for '{filename_to_convert}' ---")
    elif len(sys.argv) == 1:
        print(f"\n--- Starting bulk conversion of all photos in: '{PHOTO_STORAGE_FOLDER}' ---")
        if not os.path.exists(PHOTO_STORAGE_FOLDER):
            print(f"ERROR: Photo storage folder '{PHOTO_STORAGE_FOLDER}' does not exist. Please create it and place photos inside.")
        else:
            processed_count = 0
            for filename in os.listdir(PHOTO_STORAGE_FOLDER):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    print(f"\nProcessing: {filename}")
                    if convert_and_store_photo(filename):
                        processed_count += 1
            print(f"\n--- Finished bulk processing. Total photos successfully processed: {processed_count} ---")
    else:
        print("Usage: python convert.py <photo_filename> (for single file conversion)")
        print("   Or: python convert.py (to process all photos in data_foto folder)")
        print("Note: The user ID will be extracted from the photo_filename (without extension).")