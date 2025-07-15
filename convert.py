# convert.py - Converts Photos to Face Vectors and Stores them in Annoy Index
# This file can be triggered by watcher.py or run independently for bulk conversion.

import os
import sys
import numpy as np
import json
from annoy import AnnoyIndex
import sqlite3 # Import library SQLite

# --- Configuration ---
PHOTO_STORAGE_FOLDER = 'data_foto'
ANNOY_INDEX_PATH = 'database_foto_vector/face_vectors.ann'
ANNOY_ID_MAP_PATH = 'database_foto_vector/face_id_map.json'

# SQLite Database for User Profiles
DATABASE_USER_PROFILE = 'database_user/user_profiles.db'

# Annoy Index configuration
VECTOR_DIMENSION = 128 # The dimension of your face embeddings (simulated in this case)
METRIC = 'angular' # 'angular' for cosine similarity, 'euclidean' for Euclidean distance
N_TREES = 10 # Number of trees to build for the Annoy index

# Supported image file extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

# --- SQLite Database Functions ---
def get_user_profile_db_connection():
    """Establishes a connection to the user profiles SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_USER_PROFILE)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name (e.g., row['id'])
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
        
        # 1. Attempt to UPDATE existing user first
        cursor.execute("UPDATE users SET face_id = ? WHERE id = ?", (annoy_id, user_id))
        conn.commit()
        
        if cursor.rowcount > 0:
            # User updated successfully
            print(f"INFO: Successfully updated face_id for existing user '{user_id}' to Annoy ID '{annoy_id}'.")
            return True
        else:
            # 2. If user_id not found (no rows updated), attempt to INSERT a new record
            print(f"INFO: User ID '{user_id}' not found in database. Attempting to insert a new user record.")
            try:
                # Provide default values for name and email
                cursor.execute(
                    "INSERT INTO users (id, name, email, face_id) VALUES (?, ?, ?, ?)",
                    (user_id, f"User {user_id}", f"{user_id}@example.com", annoy_id)
                )
                conn.commit()
                print(f"INFO: Successfully inserted new user '{user_id}' with Annoy ID '{annoy_id}'.")
                return True
            except sqlite3.IntegrityError:
                # This handles cases where `id` might already exist but the initial UPDATE failed for other reasons.
                print(f"WARNING: User ID '{user_id}' already exists but could not be updated or re-inserted. Skipping.")
                return False
            
    except sqlite3.Error as e:
        print(f"ERROR: Database operation failed for user '{user_id}': {e}")
        return False
    finally:
        if conn:
            conn.close()

# --- Simulate AI Face Recognition Model for Embedding Extraction (UPDATED) ---
def extract_face_embedding(image_path):
    """
    Simulasi mendapatkan embedding wajah dari sebuah gambar secara konsisten
    berdasarkan user_id (nama file tanpa ekstensi).
    Ini penting agar 'Ronaldo' selalu menghasilkan embedding yang sama,
    dan 'Messi' selalu menghasilkan embedding yang sama, tapi berbeda satu sama lain.
    """
    user_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # Gunakan user_id sebagai seed untuk menghasilkan vektor acak yang konsisten
    if not user_id:
        seed_val = 0
    else:
        seed_val = abs(hash(user_id)) % (2**32 - 1) # Menghasilkan integer positif dari hash
    
    np.random.seed(seed_val) # Set seed untuk reproduktibilitas
    embedding = np.random.rand(VECTOR_DIMENSION).tolist()
    np.random.seed(None) # Reset seed agar operasi random lain tidak terpengaruh
    return embedding

# --- Annoy Index & Map Management Functions ---
def load_id_map():
    """Loads the Annoy ID map, or initializes it if it doesn't exist."""
    id_map = {} # Maps Annoy integer ID (string key for JSON) to user_id
    user_id_to_annoy_id = {} # Maps user_id to Annoy integer ID (for quick reverse lookups)

    if os.path.exists(ANNOY_ID_MAP_PATH):
        try:
            with open(ANNOY_ID_MAP_PATH, 'r') as f:
                id_map = json.load(f)
                # Rebuild user_id_to_annoy_id from id_map, ensuring keys are integers
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
        # Annoy index must be built before saving.
        # Check if items exist to prevent errors with empty index.
        if annoy_index.get_n_items() > 0:
            annoy_index.build(N_TREES)
            annoy_index.save(ANNOY_INDEX_PATH)
            print(f"INFO: Annoy index saved to {ANNOY_INDEX_PATH}")
        else:
            print("INFO: Annoy index is empty, skipping build and save.")

        with open(ANNOY_ID_MAP_PATH, 'w') as f:
            json.dump(id_map, f, indent=4) # Use indent for better readability of JSON
        print(f"INFO: ID map saved to {ANNOY_ID_MAP_PATH}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save Annoy index or ID map: {e}")
        return False

# --- Main Conversion Logic ---
def convert_and_store_photo(image_filename):
    """
    Takes an image filename, extracts its simulated face vector,
    stores it in the Annoy index, and updates the associated user's face_id in SQLite.
    """
    user_id = os.path.splitext(image_filename)[0] # Extract filename without extension (e.g., 'user1' from 'user1.jpg')
    image_path = os.path.join(PHOTO_STORAGE_FOLDER, image_filename)

    if not os.path.exists(image_path):
        print(f"ERROR: Photo file not found at {image_path}. Skipping conversion.")
        return False

    try:
        # Extract simulated face embedding
        face_vector = extract_face_embedding(image_path)
        if face_vector is None:
            print(f"WARNING: No valid face embedding extracted for '{image_filename}'. Skipping.")
            return False
        
        # Load existing Annoy ID map and its reverse for quick lookups
        id_map, user_id_to_annoy_id = load_id_map()

        # Create a NEW Annoy index for building/rebuilding with current and previous data
        temp_annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)

        # If an old Annoy index exists, load its items into the new temporary index
        if os.path.exists(ANNOY_INDEX_PATH):
            old_annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
            try:
                old_annoy_index.load(ANNOY_INDEX_PATH)
                # Iterate through existing items from the old index and add them to the new one.
                # Crucially, skip the current user_id's entry if it's being updated.
                for annoy_id_str, existing_user_id in id_map.items():
                    existing_annoy_id = int(annoy_id_str)
                    if existing_user_id != user_id:
                        temp_annoy_index.add_item(existing_annoy_id, old_annoy_index.get_item_vector(existing_annoy_id))
                    else:
                        print(f"INFO: Skipping old vector for user '{user_id}' as it will be updated with a new one.")
            except Exception as e:
                print(f"WARNING: Could not load existing Annoy index for rebuilding ({e}). Starting with an empty index.")
                # If loading the old index fails, reset ID maps to ensure consistency
                id_map = {}
                user_id_to_annoy_id = {}

        # Determine the Annoy integer ID for this user_id.
        # If the user_id already has an Annoy ID, reuse it. Otherwise, assign a new one.
        annoy_id = user_id_to_annoy_id.get(user_id)
        is_new_user_annoy_entry = False
        if annoy_id is None: # FIX: Corrected '==' to 'is'
            # Assign a new Annoy ID by taking the current number of items in the new index.
            # This ensures unique, sequential IDs.
            annoy_id = temp_annoy_index.get_n_items()
            is_new_user_annoy_entry = True
            
        # Add or update the item in the NEW Annoy index. Annoy's `add_item` overwrites if the ID exists.
        temp_annoy_index.add_item(annoy_id, face_vector)
        
        # Update the in-memory ID maps (both Annoy ID -> user_id and user_id -> Annoy ID)
        id_map[str(annoy_id)] = user_id
        user_id_to_annoy_id[user_id] = annoy_id

        if is_new_user_annoy_entry:
            print(f"INFO: New Annoy vector added for user_id '{user_id}' with Annoy ID '{annoy_id}'.")
        else:
            print(f"INFO: Annoy vector updated for user_id '{user_id}' with Annoy ID '{annoy_id}'.")

        # --- IMPORTANT: Update face_id in the SQLite database ---
        if update_user_face_id(user_id, annoy_id):
            print(f"INFO: Successfully linked Annoy ID '{annoy_id}' to user '{user_id}' in SQLite database.")
        else:
            print(f"ERROR: Failed to link Annoy ID '{annoy_id}' to user '{user_id}' in SQLite database.")
        # --- End SQLite Update ---

        # Finally, save the updated Annoy index and ID map to disk
        return save_annoy_index_and_map(temp_annoy_index, id_map)

    except Exception as e:
        print(f"FATAL ERROR: Processing photo '{image_filename}' failed: {e}")
        return False

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- SQLite Database Initialization ---
    # This block now ONLY ensures the 'database_user' directory exists and
    # confirms connection to 'user_profiles.db'. It assumes the 'users' table
    # with the 'face_id' column has already been created manually or by another script.
    conn = None
    try:
        # Create 'database_user' directory if it doesn't exist
        os.makedirs(os.path.dirname(DATABASE_USER_PROFILE), exist_ok=True)
        conn = sqlite3.connect(DATABASE_USER_PROFILE)
        
        # You can optionally add a check here to verify table existence if desired,
        # but it's not strictly necessary for the script to run if you're sure it exists.
        # Example check:
        # cursor = conn.cursor()
        # cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users';")
        # if cursor.fetchone() is None:
        #     print("WARNING: Table 'users' does not exist in 'user_profiles.db'. Please create it manually.")
        
        print("INFO: Confirmed connection to 'user_profiles.db'. Assuming 'users' table with 'face_id' column exists.")
    except sqlite3.Error as e:
        print(f"CRITICAL ERROR: SQLite database connection failed: {e}")
    finally:
        if conn:
            conn.close()

    # Ensure the directory for Annoy index and map files exists
    os.makedirs(os.path.dirname(ANNOY_INDEX_PATH), exist_ok=True)

    # --- Handle command-line arguments for conversion ---
    if len(sys.argv) == 2:
        # If a single filename is provided as an argument
        filename_to_convert = sys.argv[1]
        print(f"\n--- Starting conversion for single file: '{filename_to_convert}' ---")
        convert_and_store_photo(filename_to_convert)
        print(f"--- Finished conversion for '{filename_to_convert}' ---")
    elif len(sys.argv) == 1:
        # If no arguments, process all photos in the PHOTO_STORAGE_FOLDER
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
        # Invalid number of arguments
        print("Usage: python convert.py <photo_filename> (for single file conversion)")
        print("   Or: python convert.py (to process all photos in data_foto folder)")
        print("Note: The user ID will be extracted from the photo_filename (without extension).")