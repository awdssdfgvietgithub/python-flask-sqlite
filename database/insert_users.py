import sqlite3
import csv
import os
import random
import base64
import python_avatars as pa
from unidecode import unidecode
from faker import Faker
from db_scripts import connect_to_db

csv_file_path = os.path.join('models', 'potential_links.csv')

conn = connect_to_db()
locales = ["vi_VN", "en_US", "fr_FR", "es_ES", "de_DE"]

def get_user_ids(csv_file_path):
    user_ids = set()
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            node1 = row[0]
            user_ids.add(node1)
    return list(user_ids)

def get_followers(csv_file_path):
    followers = []
    with open(csv_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            user_id = row[0]
            follower_id = row[1]
            followers.append((user_id, follower_id))
    return followers

def insert_users(conn, user_ids):
    for user_id in user_ids:
        try:
            locale = random.choice(locales)
            fake = Faker(locale)

            username = f"user_{user_id}"
            display_name = fake.name()
            normal_display_name = unidecode(display_name.strip().lower())

            avatar = pa.Avatar.random()
            avatar_file_path = f"avatar_{user_id}.png"
            avatar.render(avatar_file_path)

            with open(avatar_file_path, "rb") as image_file:
                avatar_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            os.remove(avatar_file_path)

            conn.execute(
                "INSERT INTO user (id, username, password, display_name, normal_display_name, avatar) VALUES (?, ?, ?, ?, ?, ?)", 
                (user_id, username, "nostudyhaveparty", display_name, normal_display_name, avatar_base64)
            )

        except sqlite3.IntegrityError:
            pass
        except Exception as e:
            print(f"Error processing user {user_id}: {e}")

    conn.commit()
    print("Inserted users with avatars into the table successfully!")

def insert_user_followers(conn, followers):
    for user_id, follower_id in followers:
        try:
            conn.execute(
                "INSERT INTO user_followers (user_id, follower_id) VALUES (?, ?)",
                (user_id, follower_id)
            )
        except sqlite3.IntegrityError as e:
            print(f"Error inserting follower (user_id={user_id}, follower_id={follower_id}): {e}")
    conn.commit()
    print("Inserted followers into table successfully!")

def main():    
    conn = connect_to_db()

    user_ids = get_user_ids(csv_file_path)
    followers = get_followers(csv_file_path)
    
    insert_users(conn, user_ids)
    insert_user_followers(conn, followers)
    
    conn.close()
    print("Database connection closed!")

if __name__ == "__main__":
    main()