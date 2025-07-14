# convert.py - Converts Photos to Face Vectors and Stores them in Annoy Index
# This file can be triggered by watcher.py or run independently for bulk conversion.

import os
import sys
import numpy as np
import json
from annoy import AnnoyIndex

# Configuration for photo storage and Annoy index
PHOTO_STORAGE_FOLDER = 'data_foto'
ANNOY_INDEX_PATH = 'database_foto_vector/face_vectors.ann'
ANNOY_ID_MAP_PATH = 'database_foto_vector/face_id_map.json' # To map Annoy integer IDs to user_ids (filenames without extension)

# Annoy Index configuration
VECTOR_DIMENSION = 128 # Must match the dimension of face embeddings
METRIC = 'angular' # 'angular' for cosine similarity, 'euclidean' for Euclidean distance
N_TREES = 10 # Number of trees to build for the Annoy index

# Supported image file extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

# --- Simulate AI Face Recognition Model for Embedding Extraction ---
def extract_face_embedding(image_path):
    """
    Simulates extracting face embeddings from an image.
    In a real scenario, this would load a real AI model (e.g., FaceNet, ArcFace)
    and process the image to generate a feature vector.
    """
    # For demo, we will create a random vector as face representation
    return np.random.rand(VECTOR_DIMENSION).tolist() # Returns a list so it can be handled by Annoy

def load_id_map():
    """Loads the ID map, or initializes it if it doesn't exist."""
    id_map = {} # Maps Annoy integer ID to user_id (filename without extension)
    user_id_to_annoy_id = {} # Maps user_id (filename without extension) to Annoy integer ID

    if os.path.exists(ANNOY_ID_MAP_PATH):
        try:
            with open(ANNOY_ID_MAP_PATH, 'r') as f:
                id_map = json.load(f)
                # Rebuild user_id_to_annoy_id from id_map for quick lookups
                # Ensure keys from JSON are converted to int for annoy_id
                user_id_to_annoy_id = {v: int(k) for k, v in id_map.items()}
            print(f"ID map loaded from {ANNOY_ID_MAP_PATH}")
        except Exception as e:
            print(f"Error loading ID map: {e}. Initializing an empty map.")
            id_map = {}
            user_id_to_annoy_id = {}
    else:
        print("ID map not found, initializing empty map.")

    return id_map, user_id_to_annoy_id

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
    is used as the identifier. This function now handles updates by rebuilding
    the Annoy index.
    """
    user_id = os.path.splitext(image_filename)[0] # Extract filename without extension
    image_path = os.path.join(PHOTO_STORAGE_FOLDER, image_filename)

    if not os.path.exists(image_path):
        print(f"Error: Photo file not found at {image_path}")
        return False

    try:
        face_vector = extract_face_embedding(image_path)
        
        # Load existing ID map
        id_map, user_id_to_annoy_id = load_id_map()

        # Create a NEW Annoy index for building/rebuilding
        temp_annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)

        # If an old Annoy index exists, load its items and add them to the new index
        # This ensures existing vectors are preserved when rebuilding
        if os.path.exists(ANNOY_INDEX_PATH):
            old_annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
            try:
                old_annoy_index.load(ANNOY_INDEX_PATH)
                # Iterate through existing items in the old index and add them to the new one
                for annoy_id_str, existing_user_id in id_map.items():
                    existing_annoy_id = int(annoy_id_str)
                    # Only add if it's NOT the current user_id we are about to update/add
                    if existing_user_id != user_id:
                        temp_annoy_index.add_item(existing_annoy_id, old_annoy_index.get_item_vector(existing_annoy_id))
                    else:
                        print(f"Skipping old vector for '{user_id}' as it will be updated with a new one.")

            except Exception as e:
                print(f"Warning: Could not load existing Annoy index for re-building: {e}. Starting with an empty index (only new/updated item will be added).")
                # If loading fails, ensure id_map is cleared for consistency if we can't rebuild from old index
                id_map = {}
                user_id_to_annoy_id = {}


        # Determine the Annoy integer ID for this user_id
        annoy_id = user_id_to_annoy_id.get(user_id)
        is_new_user = False
        if annoy_id is None:
            # Assign a new Annoy ID if user_id is new
            # Annoy IDs must be integers. We'll use the current number of items in the *new* index as the next ID
            annoy_id = temp_annoy_index.get_n_items() # Use the current number of items as the next ID
            is_new_user = True
            
        # Add or update the item in the NEW Annoy index
        # Annoy's add_item will overwrite if the ID already exists, which is suitable for updates.
        temp_annoy_index.add_item(annoy_id, face_vector)
        
        # Update the ID map
        id_map[str(annoy_id)] = user_id # Store Annoy ID as string key for JSON serialization
        user_id_to_annoy_id[user_id] = annoy_id # Update reverse map for quick lookups

        if is_new_user:
            print(f"New face vector added for user_id '{user_id}' with Annoy ID '{annoy_id}'.")
        else:
            print(f"Face vector updated for user_id '{user_id}' with Annoy ID '{annoy_id}'.")

        # Save the updated Annoy index and ID map
        # This will build and save the temp_annoy_index
        return save_annoy_index_and_map(temp_annoy_index, id_map)

    except Exception as e:
        print(f"Error converting or storing photo '{image_filename}': {e}")
        return False

if __name__ == '__main__':
    # If run from the command line with a filename argument
    if len(sys.argv) == 2:
        filename_to_convert = sys.argv[1]
        print(f"Received conversion command for '{filename_to_convert}'")
        convert_and_store_photo(filename_to_convert)
    # If run without arguments, process all photos in the folder
    elif len(sys.argv) == 1:
        print(f"Processing all photos in folder: {PHOTO_STORAGE_FOLDER}")
        if not os.path.exists(PHOTO_STORAGE_FOLDER):
            print(f"Error: Photo storage folder '{PHOTO_STORAGE_FOLDER}' does not exist.")
        else:
            processed_count = 0
            for filename in os.listdir(PHOTO_STORAGE_FOLDER):
                if filename.lower().endswith(IMAGE_EXTENSIONS):
                    print(f"Converting and storing: {filename}")
                    if convert_and_store_photo(filename):
                        processed_count += 1
            print(f"Finished processing. Total photos processed: {processed_count}")
    else:
        print("convert.py is running. It is usually triggered by watcher.py or run for bulk conversion.")
        print("Usage: python convert.py <photo_filename> (for single file conversion)")
        print("Or: python convert.py (to process all photos in data_photo folder)")
        print("Note: The user ID will be extracted from the photo_filename (without extension).")

