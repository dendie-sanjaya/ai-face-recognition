# watcher.py - Memantau folder foto dan memicu konversi

import os
import time
import subprocess # Untuk memanggil convert.py
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sqlite3 # Untuk mendapatkan user_id dari user_profiles.db

# Konfigurasi
PHOTO_STORAGE_FOLDER = 'data_foto/data_photo'
DATABASE_USER_PROFILE = 'user_profiles.db'

# Pastikan folder foto ada
if not os.path.exists(PHOTO_STORAGE_FOLDER):
    os.makedirs(PHOTO_STORAGE_FOLDER)


class PhotoEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = os.path.basename(event.src_path)
            print(f"File baru terdeteksi: {filename}")
            
            # Asumsi: Untuk demo ini, kita akan mencoba mendapatkan user_id dari nama file
            # Dalam aplikasi nyata, user_id mungkin datang dari metadata atau proses pendaftaran
            user_id = self.get_user_id_from_filename(filename)
            if user_id:
                print(f"Memicu convert.py untuk file: {filename} dengan user_id: {user_id}")
                # Panggil convert.py sebagai subprocess
                # Gunakan sys.executable untuk memastikan interpreter Python yang benar
                subprocess.run(['python', 'convert.py', filename, user_id])
            else:
                print(f"Tidak dapat menentukan user_id untuk file: {filename}. Lewati konversi.")

if __name__ == "__main__":
    event_handler = PhotoEventHandler()
    observer = Observer()
    observer.schedule(event_handler, PHOTO_STORAGE_FOLDER, recursive=False)
    observer.start()
    print(f"Memulai pemantauan folder: {PHOTO_STORAGE_FOLDER}")
    print("Tekan Ctrl+C untuk berhenti.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("Pemantauan dihentikan.")

