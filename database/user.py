import sqlite3
from db_scripts import connect_to_db, DB_NAME

conn = connect_to_db()
print("Connected to database successfully")

conn.execute("DROP TABLE IF EXISTS user")
conn.execute("DROP TABLE IF EXISTS user_followers")
print("Drop table successfully")

conn.execute('''
CREATE TABLE IF NOT EXISTS user (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    display_name TEXT NOT NULL,
    normal_display_name TEXT NOT NULL,
    avatar TEXT
)
''')

conn.execute('''
CREATE TABLE IF NOT EXISTS user_followers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    follower_id INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES user (id) ON DELETE CASCADE,
    FOREIGN KEY (follower_id) REFERENCES user (id) ON DELETE CASCADE
)
''')

print("Created tables 'user' and 'user_followers' successfully!")

conn.close()

