# convert.py - Mengonversi Foto ke Format Vektor Wajah dan Menyimpan ke Annoy Index
# File ini akan dipicu oleh watcher.py atau dapat dijalankan secara mandiri untuk konversi bulk.

import os
import sys
import numpy as np
import json
from annoy import AnnoyIndex

# Configuration for photo storage and Annoy index
PHOTO_STORAGE_FOLDER = 'database_foto_vector/data_photo'
ANNOY_INDEX_PATH = 'database_foto_vector/face_vectors.ann'
ANNOY_ID_MAP_PATH = 'face_id_map.json' # To map Annoy integer IDs to user_ids (filenames without extension)

# Annoy Index configuration
VECTOR_DIMENSION = 128 # Must match the dimension of face embeddings
METRIC = 'angular' # 'angular' for cosine similarity, 'euclidean' for Euclidean distance
N_TREES = 10 # Number of trees to build for the Annoy index

# --- Simulate AI Face Recognition Model for Embedding Extraction ---
def extract_face_embedding(image_path):
    """
    Simulates extracting face embeddings from an image.
    In a real scenario, this would load a real AI model (e.g., FaceNet, ArcFace)
    and process the image to generate a feature vector.
    """
    # For demo, we will create a random vector as face representation
    return np.random.rand(VECTOR_DIMENSION).tolist() # Returns a list so it can be handled by Annoy

def load_annoy_index_and_map():
    """Loads the Annoy index and the ID map, or initializes them if they don't exist."""
    annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
    id_map = {} # Maps Annoy integer ID to user_id (filename without extension)
    user_id_to_annoy_id = {} # Maps user_id (filename without extension) to Annoy integer ID

    if os.path.exists(ANNOY_INDEX_PATH):
        try:
            annoy_index.load(ANNOY_INDEX_PATH)
            print(f"Annoy index loaded from {ANNOY_INDEX_PATH}")
        except Exception as e:
            print(f"Error loading Annoy index: {e}. Initializing a new one.")
            # If loading fails, create a new empty index
            annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)

    if os.path.exists(ANNOY_ID_MAP_PATH):
        with open(ANNOY_ID_MAP_PATH, 'r') as f:
            id_map = json.load(f)
            # Rebuild user_id_to_annoy_id from id_map for quick lookups
            # Ensure keys from JSON are converted to int for annoy_id
            user_id_to_annoy_id = {v: int(k) for k, v in id_map.items()}
        print(f"ID map loaded from {ANNOY_ID_MAP_PATH}")

    return annoy_index, id_map, user_id_to_annoy_id

def save_annoy_index_and_map(annoy_index, id_map):
    """Saves the Annoy index and the ID map."""
    try:
        # Annoy index must be built before saving
        # If the index is empty, build() will fail, so check if items exist
        if annoy_index.get_n_items() > 0:
            annoy_index.build(N_TREES)
            annoy_index.save(ANNOY_INDEX_PATH)
            print(f"Annoy index saved to {ANNOY_INDEX_PATH}")
        else:
            print("Annoy index is empty, skipping build and save.")

        with open(ANNOY_ID_MAP_PATH, 'w') as f:
            json.dump(id_map, f)
        print(f"ID map saved to {ANNOY_ID_MAP_PATH}")
        return True
    except Exception as e:
        print(f"Error saving Annoy index or ID map: {e}")
        return False

def convert_and_store_photo(image_filename):
    """
    Takes an image filename, extracts its face vector,
    and stores it in the Annoy index. The user_id (filename without extension)
    is used as the identifier.
    """
    user_id = os.path.splitext(image_filename)[0] # Extract filename without extension
    image_path = os.path.join(PHOTO_STORAGE_FOLDER, image_filename)

    if not os.path.exists(image_path):
        print(f"Error: Photo file not found at {image_path}")
        return False

    try:
        face_vector = extract_face_embedding(image_path)
        
        annoy_index, id_map, user_id_to_annoy_id = load_annoy_index_and_map()

        # Determine the Annoy integer ID for this user_id
        annoy_id = user_id_to_annoy_id.get(user_id)
        is_new_user = False
        if annoy_id is None:
            # Assign a new Annoy ID if user_id is new
            # Annoy IDs must be integers. We'll use the next available integer.
            annoy_id = annoy_index.get_n_items() # Use the current number of items as the next ID
            is_new_user = True
            
        # Add or update the item in the Annoy index
        # Annoy's add_item will overwrite if the ID already exists, which is suitable for updates.
        annoy_index.add_item(annoy_id, face_vector)
        
        # Update the ID map
        id_map[str(annoy_id)] = user_id # Store Annoy ID as string key for JSON serialization
        user_id_to_annoy_id[user_id] = annoy_id # Update reverse map for quick lookups

        if is_new_user:
            print(f"New face vector added for user_id '{user_id}' with Annoy ID '{annoy_id}'.")
        else:
            print(f"Face vector updated for user_id '{user_id}' with Annoy ID '{annoy_id}'.")

        # Save the updated Annoy index and ID map
        return save_annoy_index_and_map(annoy_index, id_map)

    except Exception as e:
        print(f"Error converting or storing photo '{image_filename}': {e}")
        return False

if __name__ == '__main__':
    # If run from the command line (e.g., by watcher.py)
    if len(sys.argv) == 2: # Expecting only the filename now
        filename_to_convert = sys.argv[1]
        print(f"Received conversion command for '{filename_to_convert}'")
        convert_and_store_photo(filename_to_convert)
    else:
        print("convert.py is running. It is usually triggered by watcher.py.")
        print("Usage: python convert.py <photo_filename>")
        print("Example: python convert.py rolando.jpg")
        print("Note: The user ID will be extracted from the photo_filename (without extension).")

