import sqlite3
import bcrypt
import os

# --- Constants ---
# The user database will be stored on the persistent disk
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", "/data")
DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "users.db")

def create_user_table():
    """Creates the users table in the database if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("User table created or already exists.")

def add_user(username, name, password):
    """Adds a new user to the database with a hashed password."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Hash the password
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt)
    
    try:
        cursor.execute("INSERT INTO users (username, name, password) VALUES (?, ?, ?)",
                       (username, name, hashed_password.decode('utf-8')))
        conn.commit()
        print(f"User '{username}' added successfully.")
    except sqlite3.IntegrityError:
        print(f"Error: User '{username}' already exists.")
    finally:
        conn.close()

if __name__ == "__main__":
    # This script will be run once to set up the database and add initial users.
    print(f"Database will be created at: {DB_PATH}")
    os.makedirs(PERSISTENT_DISK_PATH, exist_ok=True) # Ensure the /data directory exists
    
    create_user_table()
    
    # Add your initial test users here
    print("\nAdding initial users...")
    add_user("jsmith", "John Smith", "abc")
    add_user("rdoe", "Rebecca Doe", "def")
    
    print("\nDatabase setup complete.")
