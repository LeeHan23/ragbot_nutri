import sqlite3
import bcrypt
import os
import secrets

# --- UNIFIED PATH CONFIGURATION ---
# Get the directory of the current script
APP_DIR = os.path.dirname(os.path.abspath(__file__))
# Default to a local 'data' folder inside your project
LOCAL_DATA_PATH = os.path.join(APP_DIR, "data")
# Use the production path if the environment variable is set, otherwise use the local default
PERSISTENT_DISK_PATH = os.environ.get("PERSISTENT_DISK_PATH", LOCAL_DATA_PATH)

# --- Constants ---
DB_PATH = os.path.join(PERSISTENT_DISK_PATH, "users.db")

def create_user_table():
    """Creates the users table in the database if it doesn't exist."""
    # Ensure the directory for the database exists
    os.makedirs(PERSISTENT_DISK_PATH, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Add new columns for the one-time key and verification status
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password TEXT NOT NULL,
            verification_key TEXT,
            is_verified INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    print("User table created or already exists.")

def add_user(username, name, password):
    """Adds a new user to the database with a hashed password and a verification key."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Hash the password
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt)
    
    # Generate a secure, one-time verification key
    verification_key = secrets.token_hex(8)
    
    try:
        cursor.execute(
            "INSERT INTO users (username, name, password, verification_key) VALUES (?, ?, ?, ?)",
            (username, name, hashed_password.decode('utf-8'), verification_key)
        )
        conn.commit()
        print(f"User '{username}' added successfully. Verification key: {verification_key}")
        return verification_key
    except sqlite3.IntegrityError:
        print(f"Error: User '{username}' already exists.")
        return None
    finally:
        conn.close()

def verify_user(username, verification_key):
    """Verifies a user's account with the one-time key."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT verification_key FROM users WHERE username = ? AND is_verified = 0", (username,))
    user = cursor.fetchone()
    
    if user and user[0] == verification_key:
        cursor.execute("UPDATE users SET is_verified = 1, verification_key = NULL WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return True
    
    conn.close()
    return False

def check_login(username, password):
    """Checks user credentials against the database."""
    if not os.path.exists(DB_PATH):
        return False, None, False # DB doesn't exist

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password, name, is_verified FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        hashed_password = user[0].encode('utf-8')
        user_name = user[1]
        is_verified = bool(user[2])
        password_bytes = password.encode('utf-8')
        
        if bcrypt.checkpw(password_bytes, hashed_password):
            return True, user_name, is_verified # Return success, name, and verification status
            
    return False, None, False

if __name__ == "__main__":
    # This script can be run once to set up the database initially.
    print(f"Database will be created at: {DB_PATH}")
    create_user_table()
    print("\nDatabase setup complete. Users can now be added via the sign-up form.")
