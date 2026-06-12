import os
import sqlite3
import hashlib
import json

# Get the path to data/medhelp.db relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "medhelp.db")

def init_db():
    """Initializes the database and creates the users and test_results tables if they don't exist."""
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            sex TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create test_results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS test_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            disease TEXT NOT NULL,
            inputs TEXT NOT NULL,
            results TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hashes a password using PBKDF2-SHA256 with a unique salt."""
    salt = os.urandom(16)
    pw_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    return f"{salt.hex()}:{pw_hash.hex()}"

def verify_password(stored_password_hash: str, provided_password: str) -> bool:
    """Verifies a password against the stored PBKDF2-SHA256 hash."""
    try:
        salt_hex, hash_hex = stored_password_hash.split(':')
        salt = bytes.fromhex(salt_hex)
        pw_hash = bytes.fromhex(hash_hex)
        new_hash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return new_hash == pw_hash
    except Exception:
        return False

def create_user(email: str, password: str, name: str, age: int, sex: str) -> tuple[bool, str]:
    """Registers a new user in the database."""
    email = email.strip().lower()
    name = name.strip()
    
    if not email or not password or not name:
        return False, "Email, password, and name are required."
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Check if user already exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            return False, "An account with this email already exists."
            
        pw_hash = hash_password(password)
        cursor.execute("""
            INSERT INTO users (email, password_hash, name, age, sex)
            VALUES (?, ?, ?, ?, ?)
        """, (email, pw_hash, name, age, sex))
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except Exception as e:
        conn.close()
        return False, f"Database error: {str(e)}"

def authenticate_user(email: str, password: str) -> dict | None:
    """Checks user credentials and returns the user dict if authenticated."""
    email = email.strip().lower()
    if not email or not password:
        return None
        
    conn = sqlite3.connect(DB_PATH)
    # Configure connection to return rows as dictionaries
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        conn.close()
        
        if row and verify_password(row['password_hash'], password):
            user_dict = dict(row)
            # Remove sensitive hash before returning
            user_dict.pop('password_hash', None)
            return user_dict
        return None
    except Exception:
        conn.close()
        return None

def update_user_profile(user_id: int, name: str, age: int, sex: str) -> tuple[bool, str]:
    """Updates user profile details."""
    name = name.strip()
    if not name:
        return False, "Name cannot be empty."
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE users
            SET name = ?, age = ?, sex = ?
            WHERE id = ?
        """, (name, age, sex, user_id))
        conn.commit()
        conn.close()
        return True, "Profile updated successfully!"
    except Exception as e:
        conn.close()
        return False, f"Database error: {str(e)}"

def get_user_by_id(user_id: int) -> dict | None:
    """Retrieves user profile details by their user ID."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            user_dict = dict(row)
            user_dict.pop('password_hash', None)
            return user_dict
        return None
    except Exception:
        conn.close()
        return None

def save_test_result(user_id: int, disease: str, inputs: dict, results: dict) -> tuple[bool, str]:
    """Saves a test prediction result to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        inputs_json = json.dumps(inputs)
        results_json = json.dumps(results)
        cursor.execute("""
            INSERT INTO test_results (user_id, disease, inputs, results)
            VALUES (?, ?, ?, ?)
        """, (user_id, disease, inputs_json, results_json))
        conn.commit()
        conn.close()
        return True, "Test result saved successfully!"
    except Exception as e:
        conn.close()
        return False, f"Failed to save test result: {str(e)}"

def get_user_test_history(user_id: int) -> list[dict]:
    """Retrieves all historical test results for a user, sorted from newest to oldest."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT * FROM test_results
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """, (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            record = dict(row)
            try:
                record['inputs'] = json.loads(record['inputs'])
            except Exception:
                pass
            try:
                record['results'] = json.loads(record['results'])
            except Exception:
                pass
            history.append(record)
        return history
    except Exception:
        conn.close()
        return []

def delete_test_result(result_id: int, user_id: int) -> tuple[bool, str]:
    """Deletes a test result from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
            DELETE FROM test_results
            WHERE id = ? AND user_id = ?
        """, (result_id, user_id))
        conn.commit()
        conn.close()
        return True, "Test result deleted successfully!"
    except Exception as e:
        conn.close()
        return False, f"Failed to delete test result: {str(e)}"
