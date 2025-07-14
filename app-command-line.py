import cv2
import face_recognition
import numpy as np
from annoy import AnnoyIndex
import os
import pickle

# --- 1. Konfigurasi ---
# Path untuk menyimpan index Annoy dan mapping nama
# Pastikan folder 'database_foto_vector' ada atau akan dibuat
ANNOY_INDEX_PATH = 'database_foto_vector/face_embeddings.ann'
NAME_MAPPING_PATH = 'database_foto_vector/face_names.pkl'
# Dimensi vektor wajah (dihasilkan oleh face_recognition, biasanya 128)
VECTOR_DIMENSION = 128
# Jumlah pohon untuk Annoy (lebih banyak pohon = lebih akurat tapi lebih lambat membangun)
NUM_TREES = 10
# Metrik jarak untuk Annoy (euclidean adalah default yang baik untuk embeddings)
METRIC = 'euclidean'
# Ambang batas (threshold) untuk jarak kecocokan wajah (sesuaikan sesuai kebutuhan)
# Jarak yang lebih kecil berarti lebih mirip. Anda mungkin perlu eksperimen dengan nilai ini.
FACE_MATCH_THRESHOLD = 0.6

# --- Inisialisasi Database Vektor Annoy (Global, akan direset/dimuat) ---
annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
face_names_mapping = {}
next_id = 0

# --- Fungsi Pembantu: Memuat atau Membuat Index Annoy ---
def load_or_create_annoy_index():
    global annoy_index, face_names_mapping, next_id
    # Pastikan folder database_foto_vector ada sebelum mencoba memuat/membuat file
    os.makedirs(os.path.dirname(ANNOY_INDEX_PATH), exist_ok=True)

    if os.path.exists(ANNOY_INDEX_PATH) and os.path.exists(NAME_MAPPING_PATH):
        print(f"Memuat index Annoy dari: {ANNOY_INDEX_PATH}")
        try:
            # Annoy.load() membuat objek AnnoyIndex baru, tidak perlu menginisialisasi ulang
            temp_annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
            temp_annoy_index.load(ANNOY_INDEX_PATH)
            annoy_index = temp_annoy_index # Assign ke global variable

            with open(NAME_MAPPING_PATH, 'rb') as f:
                face_names_mapping = pickle.load(f)

            if face_names_mapping:
                next_id = max(face_names_mapping.keys()) + 1
            else:
                next_id = 0
            print(f"Index Annoy dan mapping nama berhasil dimuat. Next ID: {next_id}")
            return True # Berhasil dimuat
        except Exception as e:
            print(f"Error saat memuat index atau mapping: {e}. Membuat baru.")
            # Reset global variables jika ada error saat memuat
            annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
            face_names_mapping = {}
            next_id = 0
            return False # Gagal memuat atau tidak ada, jadi buat baru
    else:
        print("Membuat index Annoy baru.")
        # Reset global variables jika tidak ada file
        annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
        face_names_mapping = {}
        next_id = 0
        return False # Tidak ada file, jadi buat baru

# --- 2. Fungsi untuk Mendaftarkan Wajah Baru (Enrollment) ---
# Fungsi ini sekarang akan membersihkan/menginisialisasi ulang index Annoy
# sebelum menambahkan item baru. Ini perlu dilakukan karena Annoy index yang
# sudah dimuat dari file bersifat read-only.
def enroll_face(image_path, person_name, is_first_enroll=False):
    global annoy_index, face_names_mapping, next_id

    # Jika ini adalah pendaftaran pertama dalam sesi ini atau kita memulai ulang proses enrollment,
    # inisialisasi ulang AnnoyIndex dan mapping.
    # Ini mengatasi 'You can't add an item to a loaded index'
    if is_first_enroll:
        print("Menginisialisasi ulang database untuk proses pendaftaran baru...")
        annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
        face_names_mapping = {}
        next_id = 0
    else:
        # Jika bukan pendaftaran pertama, coba muat ulang index yang ada
        # untuk memastikan kita bekerja dengan versi terbaru jika ada penambahan sebelumnya.
        # Atau jika ini bukan panggilan pertama, kita tambahkan ke objek Annoy_index yang sudah ada
        # (yang akan di build ulang pada akhirnya).
        # Namun, untuk Annoy, jika sudah di-load, tidak bisa di add_item.
        # Solusi terbaik adalah: selalu menginisialisasi ulang jika ingin menambah data.
        # Atau kumpulkan semua data baru dan build sekali.
        # Untuk kesederhanaan, kita akan reset index setiap kali enroll_face dipanggil.
        # Ini berarti setiap kali Anda menjalankan enroll_face(), ia akan memulai ulang database
        # Kecuali kita mempassing flag khusus untuk menanganinya sebagai satu batch enrollment.
        pass # Logika di main loop akan menangani reset AnnoyIndex


    # Baca gambar
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Tidak dapat membaca gambar dari {image_path}")
        return False

    # Konversi gambar dari BGR (OpenCV) ke RGB (face_recognition)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Deteksi lokasi wajah
    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        print(f"Tidak ada wajah terdeteksi di {image_path}. Tidak dapat mendaftarkan.")
        return False

    # Hanya ambil wajah pertama yang terdeteksi
    face_encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]

    # Tambahkan embedding wajah ke Annoy index
    annoy_index.add_item(next_id, face_encoding)
    face_names_mapping[next_id] = person_name
    print(f"Wajah '{person_name}' (ID: {next_id}) berhasil ditambahkan ke memori.")
    next_id += 1

    # Bangun (build) index Annoy dan simpan akan dilakukan setelah semua gambar di folder diproses
    return True

# --- 3. Fungsi untuk Mengidentifikasi Wajah dari Frame (untuk Real-time dan File) ---
# Mengambil frame OpenCV (numpy array) sebagai input
def identify_face_in_frame(frame):
    # Konversi gambar dari BGR (OpenCV) ke RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi lokasi wajah dan ekstrak encoding
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    identified_faces = []

    for i, (top, right, bottom, left) in enumerate(face_locations):
        unknown_face_encoding = face_encodings[i]

        name = "Tidak Dikenal"
        match_distance = float('inf')
        similarity_percentage = 0.0 # Default 0% kemiripan

        # Penting: Periksa apakah Annoy index memiliki item sebelum mencari
        # Jika Annoy Index kosong, get_nns_by_vector akan error
        if annoy_index.get_n_items() > 0:
            # Cari tetangga terdekat di Annoy index
            nearest_ids, distances = annoy_index.get_nns_by_vector(
                unknown_face_encoding, 1, include_distances=True
            )

            if nearest_ids:
                matched_id = nearest_ids[0]
                match_distance = distances[0]

                if match_distance < FACE_MATCH_THRESHOLD:
                    name = face_names_mapping.get(matched_id, "ID Tidak Dikenal (Error)")
                    # Semakin kecil jarak dari threshold, semakin mirip
                    # Hitung persentase kemiripan: 100% saat jarak 0, 0% saat jarak sama dengan threshold
                    similarity_score = (FACE_MATCH_THRESHOLD - match_distance) / FACE_MATCH_THRESHOLD
                    similarity_percentage = min(100, max(0, similarity_score * 100))
                else:
                    name = "Tidak Dikenal (Jarak > Threshold)"
                    similarity_percentage = 0.0

        else:
            name = "Database Kosong"
            similarity_percentage = 0.0


        identified_faces.append({
            "name": name,
            "location": (top, right, bottom, left),
            "distance": match_distance,
            "similarity_percentage": similarity_percentage
        })
        # print(f"Wajah terdeteksi: '{name}', Jarak: {match_distance:.4f}, Kemiripan: {similarity_percentage:.2f}%")

    return identified_faces

# --- 4. Fungsi untuk Mengidentifikasi Wajah dari File Foto ---
def identify_face_from_file(image_path):
    print(f"\n--- Memulai Identifikasi dari File: {image_path} ---")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Tidak dapat membaca gambar dari {image_path}. Pastikan path benar dan file ada.")
        return

    # Identifikasi wajah di gambar
    identified_faces = identify_face_in_frame(image)

    # Gambar kotak dan nama di sekitar wajah yang terdeteksi
    for face in identified_faces:
        top, right, bottom, left = face['location']
        name = face['name']
        similarity_percentage = face['similarity_percentage']

        # Warna kotak: Hijau jika dikenal, Merah jika tidak dikenal/database kosong
        color = (0, 255, 0) if "Tidak Dikenal" not in name and "Database Kosong" not in name else (0, 0, 255)

        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        # Teks yang ditampilkan
        if name == "Tidak Dikenal (Jarak > Threshold)" or name == "Database Kosong":
            text_label = name
        else:
            text_label = f"{name} ({similarity_percentage:.1f}%)"

        cv2.putText(image, text_label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    # Tampilkan gambar
    cv2.imshow(f'Identifikasi Wajah dari File: {os.path.basename(image_path)}', image)
    print("Tekan tombol apapun untuk menutup jendela gambar.")
    cv2.waitKey(0) # Tunggu sampai tombol ditekan
    cv2.destroyAllWindows()
    print("Jendela gambar ditutup.")

# --- Main Program ---
if __name__ == "__main__":
    print("--- Program Identifikasi Wajah ---")

    # Pastikan folder database ada sebelum memuat atau menyimpan
    os.makedirs(os.path.dirname(ANNOY_INDEX_PATH), exist_ok=True)

    # Panggil di awal program untuk memuat database yang sudah ada
    db_loaded = load_or_create_annoy_index()
    if not db_loaded and annoy_index.get_n_items() == 0:
        print("\n[PERHATIAN]: Database wajah kosong. Anda harus mendaftarkan wajah terlebih dahulu.")
        print("Pilih '1' untuk mendaftarkan wajah.")

    while True:
        print("\n[MENU PILIHAN]")
        print("1. Daftarkan wajah baru dari folder (Enrollment)")
        print("2. Identifikasi wajah dari file foto tunggal")
        print("3. Identifikasi wajah dari Webcam (Real-time)")
        print("4. Keluar")

        choice = input("Masukkan pilihan Anda (1/2/3/4): ")

        if choice == '1':
            enrollment_folder = input("Masukkan path folder yang berisi gambar wajah untuk pendaftaran (misal: faces_to_enroll): ")
            # Pastikan folder ada
            if not os.path.isdir(enrollment_folder):
                print(f"Error: Folder '{enrollment_folder}' tidak ditemukan. Silakan buat folder ini dan masukkan gambar.")
                continue

            # --- Kunci Perbaikan untuk Error 'You can't add an item to a loaded index' ---
            # Reset Annoy Index dan mapping SEBELUM memulai proses pendaftaran
            # Ini akan memulai database baru (atau menimpa yang lama)
            print("\nMemulai proses pendaftaran. Index lama akan diinisialisasi ulang.")
            annoy_index = AnnoyIndex(VECTOR_DIMENSION, METRIC)
            face_names_mapping = {}
            next_id = 0
            # --- Akhir Perbaikan ---

            images_found = False
            for filename in os.listdir(enrollment_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(enrollment_folder, filename)
                    person_name = os.path.splitext(filename)[0] # Ambil nama file sebagai nama orang
                    # Kita tidak perlu is_first_enroll=True di enroll_face lagi karena sudah direset di sini
                    enroll_face(image_path, person_name)
                    images_found = True

            if not images_found:
                print(f"Tidak ada gambar yang ditemukan di folder '{enrollment_folder}'.")
            else:
                # Setelah semua gambar ditambahkan, BARU BANGUN DAN SIMPAN Annoy index
                if annoy_index.get_n_items() > 0:
                    annoy_index.build(NUM_TREES)
                    annoy_index.save(ANNOY_INDEX_PATH)
                    with open(NAME_MAPPING_PATH, 'wb') as f:
                        pickle.dump(face_names_mapping, f)
                    print("Index Annoy dibangun ulang dan disimpan ke disk.")
                else:
                    print("Tidak ada wajah yang berhasil didaftarkan, tidak ada index yang disimpan.")

                # Setelah pendaftaran selesai, muat ulang index untuk penggunaan identifikasi
                # Ini penting agar Annoy_index global terisi dengan data yang baru saja disimpan
                load_or_create_annoy_index()
                print("\nSelesai pendaftaran wajah dari folder. Database siap untuk identifikasi.")


        elif choice == '2':
            if annoy_index.get_n_items() == 0:
                print("Database wajah kosong. Silakan daftarkan wajah terlebih dahulu (pilihan 1).")
                continue
            image_file_path = input("Masukkan path lengkap file foto yang ingin diidentifikasi (misal: path/to/my_image.jpg): ")
            if not os.path.exists(image_file_path):
                print(f"Error: File '{image_file_path}' tidak ditemukan. Pastikan path benar.")
                continue
            identify_face_from_file(image_file_path)

        elif choice == '3':
            if annoy_index.get_n_items() == 0:
                print("Database wajah kosong. Silakan daftarkan wajah terlebih dahulu (pilihan 1).")
                continue
            print("\n--- Memulai Identifikasi dari Webcam ---")
            print("Tekan 'q' untuk keluar.")
            cap = cv2.VideoCapture(0) # 0 adalah ID webcam default Anda

            if not cap.isOpened():
                print("Error: Tidak dapat membuka webcam. Pastikan webcam terhubung dan driver terinstal.")
            else:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Gagal membaca frame dari webcam. Keluar...")
                        break

                    # Identifikasi wajah di frame
                    identified_faces = identify_face_in_frame(frame)

                    # Gambar kotak dan nama di sekitar wajah yang terdeteksi
                    for face in identified_faces:
                        top, right, bottom, left = face['location']
                        name = face['name']
                        similarity_percentage = face['similarity_percentage']

                        # Warna kotak: Hijau jika dikenal, Merah jika tidak dikenal/database kosong
                        color = (0, 255, 0) if "Tidak Dikenal" not in name and "Database Kosong" not in name else (0, 0, 255)

                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        font = cv2.FONT_HERSHEY_DUPLEX

                        # Teks yang ditampilkan
                        if name == "Tidak Dikenal (Jarak > Threshold)" or name == "Database Kosong":
                            text_label = name
                        else:
                            text_label = f"{name} ({similarity_percentage:.1f}%)"

                        cv2.putText(frame, text_label, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

                    # Tampilkan frame
                    cv2.imshow('Identifikasi Wajah Real-time', frame)

                    # Tekan 'q' untuk keluar dari loop
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
                print("Webcam ditutup. Program selesai.")

        elif choice == '4':
            print("Keluar dari program. Sampai jumpa!")
            break
        else:
            print("Pilihan tidak valid. Silakan masukkan 1, 2, 3, atau 4.")