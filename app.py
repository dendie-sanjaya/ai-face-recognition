# app.py - Backend API untuk Pengenalan Wajah dan Manajemen Profil Pengguna
# Menerima unggahan foto, melakukan komparasi menggunakan simulasi AI face recognition,
# dan mengaitkan dengan profil pengguna.

import os
import sqlite3
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np # Digunakan untuk simulasi vektor wajah
import json # Digunakan untuk menyimpan dan memuat vektor sebagai string JSON

app = Flask(__name__)

# Konfigurasi folder untuk menyimpan foto yang diunggah
UPLOAD_FOLDER = 'check_foto'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Konfigurasi database SQLite
DATABASE_USER_PROFILE = 'database_user/user_profiles.db'
DATABASE_PHOTO_VECTOR = 'database_foto_vector/face_vectors.db'

# Fungsi untuk mendapatkan koneksi database profil pengguna
def get_user_profile_db_connection():
    conn = sqlite3.connect(DATABASE_USER_PROFILE)
    conn.row_factory = sqlite3.Row # Mengembalikan baris sebagai objek mirip dict
    return conn

# Fungsi untuk mendapatkan koneksi database vektor foto
def get_photo_vector_db_connection():
    conn = sqlite3.connect(DATABASE_PHOTO_VECTOR)
    conn.row_factory = sqlite3.Row
    return conn

# --- Simulasi Model AI Pengenalan Wajah ---
# Dalam aplikasi nyata, ini akan menjadi panggilan ke model ML yang sebenarnya.
# Di sini, kita hanya membuat vektor dummy untuk tujuan demonstrasi.
def get_face_embedding(image_path):
    """
    Simulasi mendapatkan embedding wajah dari sebuah gambar.
    Dalam skenario nyata, ini akan memuat model AI (misalnya, FaceNet, ArcFace)
    dan memproses gambar untuk menghasilkan vektor fitur.
    """
    # Untuk demo, kita akan membuat vektor acak sebagai representasi wajah
    # Ukuran vektor bisa bervariasi tergantung model (misal, 128, 512)
    return np.random.rand(128).tolist() # Mengembalikan list agar bisa di-JSON-kan

def compare_face_embeddings(embedding1, embedding2, threshold=0.6):
    """
    Simulasi membandingkan dua embedding wajah.
    Dalam skenario nyata, ini akan menghitung kesamaan kosinus atau jarak Euclidean.
    """
    emb1 = np.array(embedding1)
    emb2 = np.array(embedding2)
    # Menghitung kesamaan kosinus
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity > threshold, similarity # Mengembalikan (is_match, similarity_score)

# --- Endpoint API ---

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    """
    Menerima unggahan foto, melakukan pengenalan wajah,
    dan mengaitkannya dengan profil pengguna.
    """
    if 'photo' not in request.files:
        return jsonify({"error": "Tidak ada bagian 'photo' dalam request"}), 400

    file = request.files['photo']
    if file.filename == '':
        return jsonify({"error": "Tidak ada file yang dipilih"}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath) # Simpan foto yang diunggah ke folder data_photo

        # 1. Dapatkan embedding wajah dari foto yang diunggah
        uploaded_embedding = get_face_embedding(filepath)

        # 2. Bandingkan dengan vektor wajah yang ada di database
        conn_vector = get_photo_vector_db_connection()
        cursor_vector = conn_vector.cursor()
        cursor_vector.execute("SELECT user_id, face_vector FROM photo_vectors")
        known_faces = cursor_vector.fetchall()
        conn_vector.close()

        matched_user_id = None
        highest_similarity = 0.0

        for face_data in known_faces:
            user_id = face_data['user_id']
            known_embedding = json.loads(face_data['face_vector']) # Mengurai string JSON ke list
            
            is_match, similarity = compare_face_embeddings(uploaded_embedding, known_embedding)
            
            if is_match and similarity > highest_similarity:
                highest_similarity = similarity
                matched_user_id = user_id
                # Dalam skenario nyata, Anda mungkin ingin berhenti setelah kecocokan pertama yang kuat
                # atau mencari kecocokan terbaik di antara semua.

        # 3. Gabungkan informasi profil pengguna
        if matched_user_id:
            conn_user = get_user_profile_db_connection()
            cursor_user = conn_user.cursor()
            cursor_user.execute("SELECT * FROM users WHERE id = ?", (matched_user_id,))
            user_profile = cursor_user.fetchone()
            conn_user.close()

            if user_profile:
                return jsonify({
                    "message": "Pengenalan wajah berhasil!",
                    "user_id": user_profile['id'],
                    "name": user_profile['name'],
                    "similarity_score": highest_similarity,
                    "profile_data": dict(user_profile) # Mengubah Row menjadi dict
                }), 200
            else:
                return jsonify({"message": "Wajah dikenali, tetapi profil pengguna tidak ditemukan.", "user_id": matched_user_id}), 404
        else:
            # Jika tidak ada kecocokan, Anda bisa memilih untuk:
            # a) Menyimpan wajah baru ke database vektor (untuk pendaftaran)
            # b) Meminta pengguna untuk mendaftar
            # Untuk demo ini, kita hanya akan melaporkan bahwa tidak ada kecocokan.
            return jsonify({"message": "Wajah tidak dikenali. Silakan coba lagi atau daftar."}), 404

    return jsonify({"error": "Terjadi kesalahan saat mengunggah file"}), 500

if __name__ == '__main__':
    # Pastikan database telah diinisialisasi sebelum menjalankan aplikasi
    # Anda bisa menjalankan init_db.py secara terpisah terlebih dahulu
    print(f"Aplikasi Flask berjalan. Unggah foto ke folder '{UPLOAD_FOLDER}'.")
    print("Pastikan Anda telah menjalankan init_db.py terlebih dahulu.")
    app.run(debug=True, port=5000)

