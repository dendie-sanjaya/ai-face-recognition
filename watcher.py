# watcher.py - Memantau folder foto dan memicu konversi
# Script ini memantau folder 'data_foto' untuk file gambar baru atau yang dimodifikasi.
# Ketika perubahan terdeteksi, ia akan memicu 'convert.py' untuk memproses file tersebut.

import time
import os
import sys
import subprocess
from watchdog.observers.polling import PollingObserver # Menggunakan PollingObserver untuk kompatibilitas yang lebih luas (termasuk WSL)
from watchdog.events import FileSystemEventHandler

# --- Konfigurasi ---
# Folder yang akan diawasi untuk file gambar baru/dimodifikasi
# Pastikan ini adalah jalur yang benar ke folder 'data_foto' di mana gambar-gambar Anda disimpan.
# Contoh: '/mnt/c/Users/NamaAnda/ProyekAnda/data_foto' jika di WSL dan proyek ada di C:
# Atau 'data_foto' jika watcher.py berada di direktori root proyek yang sama dengan data_foto.
WATCH_DIRECTORY = 'data_foto'

# Script yang akan dijalankan ketika ada perubahan file gambar terdeteksi
CONVERT_SCRIPT = "convert.py"

# Ekstensi file gambar yang didukung
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

# --- Pastikan folder yang diawasi ada ---
if not os.path.exists(WATCH_DIRECTORY):
    os.makedirs(WATCH_DIRECTORY)
    print(f"INFO: Direktori '{WATCH_DIRECTORY}' dibuat.")

# --- DEBUG: Cetak jalur absolut saat startup ---
# Menggunakan string literal 'python3' karena itu yang akan dieksekusi secara eksplisit
print(f"DEBUG: DIRECTORY YANG DIAMATI (ABSOLUT): {os.path.abspath(WATCH_DIRECTORY)}")
print(f"DEBUG: SCRIPT KONVERSI: {CONVERT_SCRIPT}")
print(f"DEBUG: PYTHON EXECUTABLE YANG DIGUNAKAN: python3 (eksekusi langsung)") # Perubahan di sini
# --- Akhir DEBUG ---

class PhotoEventHandler(FileSystemEventHandler):
    def __init__(self, watch_dir, convert_script):
        self.watch_dir = watch_dir
        self.convert_script = convert_script
        # Dictionary untuk menyimpan timestamp modifikasi terakhir dari SETIAP file yang diproses
        # Ini mencegah reprocessing berulang untuk event yang sama (misalnya, on_created diikuti on_modified)
        self.processed_files_mtimes = {} 
        print(f"Watchdog: Mengawasi direktori: {self.watch_dir}")
        print(f"Watchdog: Akan memicu '{self.convert_script}' untuk setiap file gambar baru/termodifikasi.")

    def _process_file(self, file_path):
        """
        Fungsi helper untuk memproses file, hanya jika itu adalah file gambar.
        Ini juga menangani deduplikasi event berdasarkan timestamp modifikasi.
        """
        # Filter untuk memastikan hanya file gambar yang diproses
        if not file_path.lower().endswith(IMAGE_EXTENSIONS):
            print(f"DEBUG: Melewatkan file non-gambar: {file_path}")
            return

        # Pastikan file ada sebelum mencoba mendapatkan mtime (penting jika file dihapus terlalu cepat)
        if not os.path.exists(file_path):
            print(f"DEBUG: File {file_path} tidak ditemukan saat event. Melewatkan.")
            return

        current_mtime = os.path.getmtime(file_path)

        # Cek apakah file sudah diproses baru-baru ini berdasarkan timestamp modifikasi
        if file_path in self.processed_files_mtimes and self.processed_files_mtimes[file_path] == current_mtime:
            print(f"DEBUG: Melewatkan {file_path} karena sudah diproses baru-baru ini (mtime sama).")
            return

        print(f"\n--- Watchdog mendeteksi perubahan pada file: {file_path} ---")
        
        # Tambahkan sedikit jeda untuk memastikan file selesai ditulis/disimpan ke disk
        # Ini penting terutama untuk file besar atau saat disalin dari sumber lain.
        time.sleep(1) 

        # Dapatkan hanya nama file (misalnya, 'ronaldo.jpg')
        # 'convert.py' mengharapkan ini sebagai argumennya.
        filename_only = os.path.basename(file_path)

        try:
            # Jalankan convert.py sebagai subprocess dengan nama file sebagai argumen
            # Menggunakan 'python3' secara eksplisit
            command = ['python3', self.convert_script, filename_only] # Perubahan di sini
            
            # --- DEBUG: Cetak perintah yang akan dieksekusi ---
            print(f"DEBUG: Perintah yang akan dieksekusi: {' '.join(command)}")
            print(f"DEBUG: Memulai subprocess '{self.convert_script}'...")
            # --- Akhir DEBUG ---

            # Jalankan subprocess dan tangkap outputnya
            process = subprocess.run(command, capture_output=True, text=True, check=True)
            
            # --- DEBUG: Output dari convert.py ---
            print(f"--- Output dari {self.convert_script} ---")
            print(process.stdout)
            if process.stderr:
                print(f"--- Error dari {self.convert_script} ---")
                print(process.stderr)
            print(f"--- {self.convert_script} selesai ---")
            # --- Akhir DEBUG ---

            # Setelah berhasil diproses, catat timestamp modifikasi terakhir
            self.processed_files_mtimes[file_path] = current_mtime
            print(f"INFO: Konversi untuk '{filename_only}' berhasil diselesaikan oleh {self.convert_script}.")

        except subprocess.CalledProcessError as e:
            print(f"ERROR: '{self.convert_script}' gagal untuk '{filename_only}'. Return code: {e.returncode}")
            print(f"STDERR:\n{e.stderr}")
        except FileNotFoundError:
            print(f"ERROR: Interpreter 'python3' atau script '{self.convert_script}' tidak ditemukan di path yang ditentukan.") # Perubahan di sini
            print(f"Pastikan 'python3' ada di PATH sistem Anda, dan '{self.convert_script}' ada di direktori yang sama atau di PATH.")
        except Exception as e:
            print(f"ERROR: Terjadi kesalahan tidak terduga saat menjalankan '{self.convert_script}' untuk '{filename_only}': {e}")
            import traceback
            traceback.print_exc() # Cetak stack trace lengkap untuk debug lebih lanjut

    def on_created(self, event):
        """Dipanggil ketika sebuah file atau direktori dibuat."""
        if not event.is_directory:
            print(f"DEBUG: Event on_created terpicu untuk: {event.src_path}")
            self._process_file(event.src_path)

    def on_modified(self, event):
        """Dipanggil ketika sebuah file atau direktori dimodifikasi."""
        if not event.is_directory:
            print(f"DEBUG: Event on_modified terpicu untuk: {event.src_path}")
            self._process_file(event.src_path)
    
    # Anda juga bisa menambahkan on_deleted atau on_moved jika perlu
    # def on_deleted(self, event):
    #     if not event.is_directory:
    #         print(f"INFO: File dihapus: {event.src_path}. Pertimbangkan untuk menghapus dari Annoy Index.")
    #         # Logika untuk menghapus dari Annoy Index dan DB jika diperlukan
    #         # Misalnya: subprocess.run(['python', 'delete_from_annoy.py', os.path.basename(event.src_path)])
    #         if event.src_path in self.processed_files_mtimes:
    #             del self.processed_files_mtimes[event.src_path]

if __name__ == "__main__":
    event_handler = PhotoEventHandler(WATCH_DIRECTORY, CONVERT_SCRIPT)
    observer = PollingObserver() # Menggunakan PollingObserver
    
    # recursive=False untuk hanya mengawasi folder utama (WATCH_DIRECTORY), bukan subfolder di dalamnya.
    # Jika Anda ingin mengawasi subfolder, ubah ini menjadi recursive=True.
    observer.schedule(event_handler, WATCH_DIRECTORY, recursive=False) 
    observer.start()
    print(f"\nMemulai pemantauan folder: {WATCH_DIRECTORY}")
    print("Tekan Ctrl+C untuk berhenti.")
    try:
        while True:
            time.sleep(1) # Interval polling default untuk PollingObserver adalah 1 detik
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    print("Watchdog berhenti.")
