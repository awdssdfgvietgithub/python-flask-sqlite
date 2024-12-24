import sqlite3

DB_NAME = f'C://Users//Innotech_mobile13//Documents//Huit//social_network//python-flask-sqlite//database//database.db'

def connect_to_db():
    try:
        conn = sqlite3.connect(DB_NAME)
        print("Connected to database successfully")
        return conn
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return None