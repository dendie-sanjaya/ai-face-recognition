# init_db.py - Initializes the user database

import sqlite3
import os

# Database file configuration
DATABASE_USER_PROFILE = 'database_user/user_profiles.db'

def init_user_profile_db():
    """
    Initializes the user_profiles.db database with the 'users' table.
    The 'id' column will be used to store the photo filename (without extension) as the key.
    """
    conn = sqlite3.connect(DATABASE_USER_PROFILE)
    cursor = conn.cursor()

    # Create the users table if it doesn't exist
    # The 'id' column will now store the photo filename (without extension) as the PRIMARY KEY
    # Adding new columns: date_of_birth, height, weight, playing_experience, residence, blood_type
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, -- This will store the photo filename (without extension) as the key
            face_id INTEGER UNIQUE, -- Unique ID for the face vector in Annoy index
            name TEXT NOT NULL,
            email TEXT UNIQUE,
            date_of_birth TEXT,
            height REAL, -- Using REAL for decimal values (e.g., height in cm)
            weight REAL, -- New column for weight (berat badan) in kg
            playing_experience TEXT,
            residence TEXT,
            blood_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print(f"User profile database '{DATABASE_USER_PROFILE}' initialized.")

# The init_photo_vector_db function has been removed as requested.
# Vectors will be stored in Annoy Index, not in the SQLite database.

def add_initial_profiles():
    """
    Adds 3 initial profiles (Rolando, Messi, Marcelino) to the user database
    along with additional information.
    User IDs will be simulated as photo filenames (without extension).
    NOTE: In a real implementation with Annoy, these profiles might be
    added after their photos are processed and indexed.
    """
    conn = sqlite3.connect(DATABASE_USER_PROFILE)
    cursor = conn.cursor()

    profiles = [
        {
            "id": "20250001", # Changed from "rolando.jpg"
            "name": "Rolando",
            "email": "rolando@example.com",
            "date_of_birth": "1985-02-05",
            "height": 187.0,
            "weight": 83.0, # Weight in kg
            "playing_experience": "Professional Football",
            "residence": "Lisbon, Portugal",
            "blood_type": "A+"
        },
        {
            "id": "20250002", # Changed from "messi.jpg"
            "name": "Messi",
            "email": "messi@example.com",
            "date_of_birth": "1987-06-24",
            "height": 170.0,
            "weight": 72.0, # Weight in kg
            "playing_experience": "Professional Football",
            "residence": "Paris, France",
            "blood_type": "A-"
        },
        {
            "id": "20250003", # Changed from "marcelino.jpg"
            "name": "Marcelino",
            "email": "marcelino@example.com",
            "date_of_birth": "1990-01-15",
            "height": 175.0,
            "weight": 68.0, # Weight in kg
            "playing_experience": "Amateur Basketball",
            "residence": "Jakarta, Indonesia",
            "blood_type": "O+"
        }
    ]

    for profile in profiles:
        try:
            cursor.execute(
                """INSERT INTO users (
                    id, name, email, date_of_birth, height, weight,
                    playing_experience, residence, blood_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile["id"], profile["name"], profile["email"],
                    profile["date_of_birth"], profile["height"], profile["weight"],
                    profile["playing_experience"], profile["residence"],
                    profile["blood_type"]
                )
            )
            print(f"Profile '{profile['name']}' with ID (filename without extension) '{profile['id']}' added.")
        except sqlite3.IntegrityError:
            print(f"Profile '{profile['name']}' with ID (filename without extension) '{profile['id']}' already exists.")
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    # Delete old database if it exists to start fresh (optional, for development)
    # This will delete all existing data in the database
    # if os.path.exists(DATABASE_USER_PROFILE):
    #     os.remove(DATABASE_USER_PROFILE)
    #     print(f"Deleting old database: {DATABASE_USER_PROFILE}")

    init_user_profile_db()
    add_initial_profiles()
    print("Database initialization complete.")
    print("You now need to modify app.py, convert.py, and watcher.py")
    print("to use Annoy Index and manage filenames (without extension) as user keys.")

