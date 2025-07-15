# app.py - Backend API untuk Pengenalan Wajah dan Manajemen Profil Pengguna
# Menerima unggahan foto, melakukan komparasi menggunakan simulasi AI face recognition
# dengan Annoy Index, dan mengaitkan dengan profil pengguna.

import os
import sqlite3
import numpy as np
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from annoy import AnnoyIndex # Import AnnoyIndex

app = Flask(__name__)

# Konfigurasi folder untuk menyimpan foto yang diunggah
UPLOAD_FOLDER = 'check_foto'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Konfigurasi database SQLite untuk profil pengguna
DATABASE_USER_PROFILE = 'database_user/user_profiles.db'

# Konfigurasi Annoy Index
ANNOY_INDEX_PATH = 'database_foto_vector/face_vectors.ann' # Path ke file Annoy index
ANNOY_ID_MAP_PATH = 'database_foto_vector/face_id_map.json' # Path ke file JSON pemetaan ID Annoy ke user_id (nama file tanpa ekstensi)

# Annoy Index configuration (harus konsisten dengan convert.py)
VECTOR_DIMENSION = 128
METRIC = 'angular' # 'angular' for cosine similarity, 'euclidean' for Euclidean distance

# --- Fungsi untuk mendapatkan koneksi database profil pengguna ---
def get_user_profile_db_connection():
    conn = sqlite3.connect(DATABASE_USER_PROFILE)
    conn.row_factory = sqlite3.Row # Mengembalikan baris sebagai objek mirip dict
    return conn

# --- Simulasi Model AI Pengenalan Wajah (DIREVISI untuk konsistensi) ---
def get_face_embedding(image_path):
    """
    Simulasi mendapatkan embedding wajah dari sebuah gambar secara konsisten
    berdasarkan user_id (nama file tanpa ekstensi).
    Ini penting agar 'Ronaldo' selalu menghasilkan embedding yang sama,
    dan 'Messi' selalu menghasilkan embedding yang sama, tapi berbeda satu sama lain.
    """
    user_id = os.path.splitext(os.path.basename(image_path))[0]
    
    # Gunakan user_id sebagai seed untuk menghasilkan vektor acak yang konsisten
    # Konversi user_id (string) ke integer untuk seed. Hash bisa menjadi pilihan.
    # Misalnya, CRC32 hash dari string.
    # Pastikan user_id tidak kosong
    if not user_id:
        seed_val = 0
    else:
        seed_val = abs(hash(user_id)) % (2**32 - 1) # Menghasilkan integer positif dari hash
    
    np.random.seed(seed_val) # Set seed untuk reproduktibilitas
    embedding = np.random.rand(VECTOR_DIMENSION).tolist()
    np.random.seed(None) # Reset seed agar operasi random lain tidak terpengaruh
    return embedding

def load_annoy_index_and_map():
    """Loads the Annoy index and the ID map."""
    annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
    id_map = {}

    if os.path.exists(ANNOY_INDEX_PATH):
        try:
            annoy_index.load(ANNOY_INDEX_PATH)
            print(f"INFO: Annoy index loaded from {ANNOY_INDEX_PATH}")
        except Exception as e:
            print(f"ERROR: Error loading Annoy index from {ANNOY_INDEX_PATH}: {e}. Please ensure it's built by convert.py.")
            return None, None # Return None if loading fails

    if os.path.exists(ANNOY_ID_MAP_PATH):
        try:
            with open(ANNOY_ID_MAP_PATH, 'r') as f:
                id_map = json.load(f)
            print(f"INFO: ID map loaded from {ANNOY_ID_MAP_PATH}")
        except Exception as e:
            print(f"ERROR: Error loading ID map from {ANNOY_ID_MAP_PATH}: {e}. Please ensure it's built by convert.py.")
            return None, None # Return None if map loading fails
    else:
        print(f"WARNING: ID map not found at {ANNOY_ID_MAP_PATH}.")
        return None, None # Return None if map not found

    return annoy_index, id_map

# --- Endpoint API ---

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    """
    Menerima unggahan foto, melakukan pengenalan wajah menggunakan Annoy Index,
    dan mengaitkannya dengan profil pengguna.
    """
    if 'photo' not in request.files:
        return jsonify({"error": "No 'photo' part in the request"}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath) # Simpan foto yang diunggah ke folder check_foto

        # 1. Dapatkan embedding wajah dari foto yang diunggah
        uploaded_embedding = get_face_embedding(filepath)

        # 2. Muat Annoy Index dan ID Map
        annoy_index, id_map = load_annoy_index_and_map()
        if annoy_index is None or id_map is None:
            return jsonify({"error": "Face recognition system is not ready. Please run convert.py first to build the Annoy index and ID map."}), 503

        # 3. Bandingkan dengan vektor wajah yang ada di Annoy Index
        # Cari 1 tetangga terdekat dan jaraknya
        
        # Threshold untuk cosine similarity (0.0 sampai 1.0)
        # Angka yang lebih tinggi berarti lebih mirip.
        # Anda mungkin perlu menyesuaikan nilai ini di dunia nyata.
        SIMILARITY_THRESHOLD = 0.75 # Misalnya, 75% kemiripan kosinus

        matched_user_id = None
        highest_similarity = 0.0
        matched_annoy_id = None # Tambahkan untuk menyimpan Annoy ID yang cocok
        raw_distance = None     # Tambahkan untuk menyimpan jarak mentah dari Annoy

        nearest_neighbors_with_distances = annoy_index.get_nns_by_vector(
            uploaded_embedding, 1, include_distances=True
        )

        if nearest_neighbors_with_distances and nearest_neighbors_with_distances[0]:
            annoy_ids_list = nearest_neighbors_with_distances[0]
            distances_list = nearest_neighbors_with_distances[1]

            if annoy_ids_list:
                closest_annoy_id = annoy_ids_list[0]
                closest_distance = distances_list[0]
                
                # Simpan raw_distance
                raw_distance = closest_distance

                # Konversi jarak Annoy (angular) ke cosine similarity
                # Rumus: cosine_similarity = 1 - (distance^2 / 2)
                # Pastikan distance tidak melebihi sqrt(2) untuk menghindari masalah dengan nilai negatif
                if METRIC == 'angular':
                    capped_distance_sq = min(closest_distance**2, 2.0)
                    similarity = 1 - (capped_distance_sq / 2.0)
                elif METRIC == 'euclidean':
                    # Untuk Euclidean, umumnya similarity = 1 / (1 + distance) atau menggunakan batas max_distance - distance
                    # Karena ini simulasi, kita bisa improvisasi:
                    # Semakin kecil jarak, semakin tinggi kemiripan. Kita bisa normalisasi jika ada max_distance.
                    # Untuk tujuan demo ini, kita biarkan saja sebagai contoh.
                    # Anda mungkin ingin mendefinisikan max_possible_distance dan then (max_possible_distance - closest_distance) / max_possible_distance
                    # Untuk saat ini, kita akan simulasikan metrik angular saja untuk contoh.
                    similarity = 1 / (1 + closest_distance) # Ini hanya contoh sederhana, bukan konversi standar
                else:
                    similarity = 0.0 # Metric not handled

                print(f"DEBUG: Closest Annoy ID: {closest_annoy_id}, Distance: {closest_distance:.4f}, Cosine Similarity: {similarity:.4f}")

                if similarity >= SIMILARITY_THRESHOLD:
                    # Dapatkan user_id dari id_map
                    matched_user_id = id_map.get(str(closest_annoy_id))
                    highest_similarity = similarity
                    matched_annoy_id = closest_annoy_id # Simpan Annoy ID yang cocok
                else:
                    print(f"INFO: Similarity {similarity:.4f} is below threshold {SIMILARITY_THRESHOLD:.2f}.")
            else:
                print("INFO: Annoy found no nearest neighbors for the uploaded embedding.")
        else:
            print("INFO: Annoy's get_nns_by_vector returned empty results.")

        # 4. Gabungkan informasi profil pengguna dari SQLite (user_profiles.db)
        if matched_user_id:
            conn_user = get_user_profile_db_connection()
            cursor_user = conn_user.cursor()
            # Gunakan 'id' yang merupakan nama file tanpa ekstensi
            cursor_user.execute("SELECT * FROM users WHERE id = ?", (matched_user_id,))
            user_profile = cursor_user.fetchone()
            conn_user.close()

            if user_profile:
                return jsonify({
                    "message": "Face recognition successful!",
                    "user_id": user_profile['id'],
                    "name": user_profile['name'],
                    "face_id_in_annoy": matched_annoy_id, # Annoy ID yang cocok
                    "similarity_score": highest_similarity,
                    "raw_annoy_distance": raw_distance, # Jarak mentah dari Annoy
                    "threshold_used": SIMILARITY_THRESHOLD,
                    "profile_data": dict(user_profile) # Mengubah Row menjadi dict
                }), 200
            else:
                return jsonify({
                    "message": f"Face recognized (User ID: {matched_user_id}), but user profile not found in the database.",
                    "user_id": matched_user_id,
                    "face_id_in_annoy": matched_annoy_id,
                    "similarity_score": highest_similarity,
                    "raw_annoy_distance": raw_distance
                }), 404
        else:
            return jsonify({
                "message": "Face not recognized or similarity below threshold. Please try again or register.",
                "uploaded_filename": filename,
                "highest_similarity_found": highest_similarity,
                "raw_annoy_distance_found": raw_distance,
                "threshold_used": SIMILARITY_THRESHOLD
            }), 404

    return jsonify({"error": "An error occurred during file upload."}), 500

if __name__ == '__main__':
    print(f"Flask application running. Upload photos to folder '{UPLOAD_FOLDER}'.")
    print("Ensure you have created 'database_user/user_profiles.db' and run 'convert.py' to build the Annoy Index and populate the database first.")
    app.run(debug=True, port=5000)