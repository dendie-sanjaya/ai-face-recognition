# app.py - Backend API untuk Pengenalan Wajah dan Manajemen Profil Pengguna

import os
import sqlite3
import numpy as np
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from annoy import AnnoyIndex
import cv2 # Import OpenCV
from mtcnn.mtcnn import MTCNN # For face detection
from keras_facenet import FaceNet # For FaceNet model

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
PHOTO_STORAGE_FOLDER = 'data_foto' # Folder untuk foto referensi
ANNOY_INDEX_PATH = 'database_foto_vector/face_vectors.ann'
ANNOY_ID_MAP_PATH = 'database_foto_vector/face_id_map.json'
DATABASE_USER_PROFILE = 'database_user/user_profiles.db'

# Annoy Index configuration
VECTOR_DIMENSION = 512 # Tetap 512 untuk FaceNet
METRIC = 'angular'
N_TREES = 10 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max 16 MB

# Allowed image file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff'}

# --- Inisialisasi Model FaceNet dan MTCNN secara Global ---
# Model ini akan dimuat sekali saat program dimulai
print("INFO: Memuat model FaceNet dan MTCNN untuk app.py...")
try:
    facenet_model = FaceNet()
    mtcnn_detector = MTCNN()
    print("INFO: Model FaceNet dan MTCNN berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat model FaceNet atau MTCNN: {e}")
    print("Pastikan TensorFlow dan pustaka terkait sudah terinstal dengan benar.")
    import sys
    sys.exit(1) # Keluar dari program jika model tidak bisa dimuat

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Fungsi Ekstraksi Embedding Wajah Menggunakan FaceNet ---
def get_face_embedding(image_path):
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
        
        # Asumsi hanya ada satu wajah utama yang ingin dikenali (wajah pertama yang terdeteksi)
        # Jika Anda ingin menangani multiple faces, logika ini perlu diubah
        x, y, width, height = faces[0]['box']
        
        # Ekstrak wajah dari gambar
        x1, y1 = abs(x), abs(y)
        x2, y2 = abs(x) + width, abs(y) + height
        face_img = img[y1:y2, x1:x2]

        # Ubah ukuran wajah ke ukuran input yang dibutuhkan FaceNet (160x160)
        face_img = cv2.resize(face_img, (160, 160))
        
        # Ekstraksi embedding menggunakan FaceNet
        embedding = facenet_model.embeddings([face_img])[0]
        
        return embedding.tolist() # Kembalikan sebagai list

    except Exception as e:
        print(f"ERROR: Gagal mengekstrak embedding dari '{image_path}': {e}")
        return None

# --- Fungsi Manajemen Database SQLite ---
def get_db_connection():
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_USER_PROFILE)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        print(f"ERROR: Gagal koneksi ke database {DATABASE_USER_PROFILE}: {e}")
    return conn

# --- Annoy Index & Map Management Functions ---
def load_annoy_index_and_map():
    annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
    id_map = {}
    try:
        if os.path.exists(ANNOY_INDEX_PATH):
            annoy_index.load(ANNOY_INDEX_PATH)
            with open(ANNOY_ID_MAP_PATH, 'r') as f:
                id_map = json.load(f)
            print("INFO: Annoy index dan ID map berhasil dimuat.")
        else:
            print("WARNING: Annoy index atau ID map tidak ditemukan. Pastikan convert.py sudah dijalankan.")
            return None, None
    except Exception as e:
        print(f"ERROR: Gagal memuat Annoy index atau ID map: {e}")
        return None, None
    return annoy_index, id_map

# --- API Endpoints ---
@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({"error": "No photo part in the request"}), 400
    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "No selected photo"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        uploaded_embedding = get_face_embedding(filepath)

        if uploaded_embedding is None:
            # Hapus file yang diunggah jika tidak ada wajah terdeteksi
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Gagal mendapatkan embedding wajah dari foto yang diunggah. Pastikan gambar berisi wajah yang jelas."}), 400

        annoy_index, id_map = load_annoy_index_and_map()
        if annoy_index is None or id_map is None:
            # Hapus file yang diunggah
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({"error": "Sistem pengenalan wajah belum siap. Harap jalankan convert.py terlebih dahulu untuk membangun indeks Annoy dan peta ID."}), 503

        # UBAH THRESHOLD DI SINI
        SIMILARITY_THRESHOLD = 0.6 

        matched_user_id = None
        highest_similarity = 0.0
        matched_annoy_id = None
        raw_distance = None

        # Periksa apakah indeks memiliki item sebelum mencari
        if annoy_index.get_n_items() > 0:
            nearest_neighbors_with_distances = annoy_index.get_nns_by_vector(
                uploaded_embedding, 1, include_distances=True
            )

            if nearest_neighbors_with_distances and nearest_neighbors_with_distances[0]:
                closest_annoy_id = nearest_neighbors_with_distances[0][0]
                closest_distance = nearest_neighbors_with_distances[1][0]
                
                raw_distance = closest_distance

                # Konversi jarak Annoy (angular) ke cosine similarity
                # Formula: cosine_similarity = 1 - (distance^2 / 2)
                capped_distance_sq = min(closest_distance**2, 2.0) # Batasi untuk menghindari masalah floating point
                similarity = 1 - (capped_distance_sq / 2.0)

                print(f"DEBUG: Closest Annoy ID: {closest_annoy_id}, Distance: {closest_distance:.4f}, Cosine Similarity: {similarity:.4f}")

                if similarity >= SIMILARITY_THRESHOLD:
                    matched_user_id = id_map.get(str(closest_annoy_id))
                    highest_similarity = similarity
                    matched_annoy_id = closest_annoy_id 
                else:
                    print(f"INFO: Similarity {similarity:.4f} is below threshold {SIMILARITY_THRESHOLD:.2f}.")
            else:
                print("INFO: Annoy found no nearest neighbors for the uploaded embedding.")
        else:
            print("INFO: Annoy index is empty. No faces to compare against.")

        # Hapus file yang diunggah setelah diproses (baik berhasil dikenali atau tidak)
        if os.path.exists(filepath):
            os.remove(filepath)

        if matched_user_id:
            conn = get_db_connection()
            user_profile = None
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE id = ?", (matched_user_id,))
                user_profile = cursor.fetchone()
                conn.close()

            if user_profile:
                return jsonify({
                    "message": "Face recognition successful!",
                    "user_id": user_profile['id'],
                    "name": user_profile['name'],
                    "face_id_in_annoy": matched_annoy_id, 
                    "similarity_score": highest_similarity,
                    "raw_annoy_distance": raw_distance, 
                    "threshold_used": SIMILARITY_THRESHOLD,
                    "profile_data": dict(user_profile) 
                }), 200
            else:
                return jsonify({
                    "message": "Recognized face, but user profile not found in database. Make sure user data exists in your separate registration process.",
                    "user_id": matched_user_id,
                    "similarity_score": highest_similarity
                }), 404
        else:
            return jsonify({
                "message": "Face not recognized or similarity below threshold. No match found in the database. Please try again or ensure the person is registered.",
                "uploaded_filename": filename,
                "highest_similarity_found": highest_similarity,
                "raw_annoy_distance_found": raw_distance,
                "threshold_used": SIMILARITY_THRESHOLD
            }), 404

    # Jika file tidak diizinkan atau ada kesalahan lain sebelum pemrosesan embedding
    return jsonify({"error": "Terjadi kesalahan saat mengunggah file atau format tidak didukung."}), 500

# Endpoint untuk mengambil daftar semua user dari database (tetap ada karena berguna untuk debugging)
@app.route('/users', methods=['GET'])
def get_users():
    conn = get_db_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, face_id FROM users")
        users = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify(users), 200
    return jsonify({"error": "Gagal koneksi ke database."}), 500

# Direktori unggahan akan dibuat saat aplikasi dijalankan jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(os.path.dirname(ANNOY_INDEX_PATH)):
    os.makedirs(os.path.dirname(ANNOY_INDEX_PATH))
if not os.path.exists(os.path.dirname(DATABASE_USER_PROFILE)):
    os.makedirs(os.path.dirname(DATABASE_USER_PROFILE))

# CATATAN: @app.before_request initialize_db_table() telah dihapus
# karena diasumsikan manajemen tabel dilakukan di luar app.py

if __name__ == '__main__':
    app.run(debug=True, port=5000)