import sqlite3
import os

DB_NAME = os.path.join('database', 'database.db')

def connect_to_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        print("Connected to database successfully")
        return conn
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return None