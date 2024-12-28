import sqlite3
from unidecode import unidecode
from flask import Flask, request, jsonify
from database.db_scripts import connect_to_db
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from tqdm import tqdm

app = Flask(__name__)

project_path = os.path.join("models")
processed_graph_path = os.path.join(project_path, "processed_graph.graphml")
model_checkpoint = os.path.join(project_path, "ML_RandomForest_model.pkl")
node2vec_full_model_checkpoint_path = os.path.join(project_path, "node2vec_full_model.pkl")
nodes_features_checkpoint_path = os.path.join(project_path, "nodes_features.pkl")

# API: Login
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "fail", "message": "Request body is missing"}), 400

        username = data.get('username')
        password = data.get('password')

        if not username or not password:
            return jsonify({"status": "fail", "message": "Username and password are required"}), 400

        conn = connect_to_db()
        if not conn:
            return jsonify({"status": "fail", "message": "Failed to connect to database"}), 500

        cursor = conn.execute(
            "SELECT id, username, display_name, avatar FROM user WHERE username = ? AND password = ?",
            (username, password)
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            return jsonify({
                "status": "success",
                "user": {
                    "id": user[0],
                    "username": user[1],
                    "display_name": user[2],
                    "avatar": user[3],
                }
            })
        else:
            return jsonify({"status": "fail", "message": "Invalid username or password"}), 401

    except sqlite3.Error as e:
        return jsonify({"status": "fail", "message": f"Database error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"status": "fail", "message": f"An unexpected error occurred: {str(e)}"}), 500

# API: Get info of a user
@app.route('/get_info_user/<int:user_id>', methods=['GET'])
def get_info_user(user_id):
    try:
        conn = connect_to_db()
        if not conn:
            return jsonify({"status": "fail", "message": "Failed to connect to database"}), 500

        cursor = conn.execute(
            """
            SELECT 
                u.id, u.username, u.avatar, u.display_name,
                (SELECT COUNT(*) FROM user_followers WHERE user_id = u.id) AS followers,
                (SELECT COUNT(*) FROM user_followers WHERE follower_id = u.id) AS following
            FROM user u WHERE u.id = ?
            """,
            (user_id,)
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            return jsonify({
                "status": "success",
                "user": {
                    "id": user[0],
                    "username": user[1],
                    "avatar": user[2],
                    "display_name": user[3],
                    "followers": user[4],
                    "following": user[5]
                }
            })
        else:
            return jsonify({"status": "fail", "message": "User not found"}), 404

    except sqlite3.Error as e:
        return jsonify({"status": "fail", "message": f"Database error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"status": "fail", "message": f"An unexpected error occurred: {str(e)}"}), 500

# API: Get all users
@app.route('/get_all_user', methods=['POST'])
def get_all_user():
    try:
        data = request.get_json()
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)

        if page < 1 or per_page < 1:
            return jsonify({"status": "fail", "message": "Page and per_page must be greater than 0"}), 400

        offset = (page - 1) * per_page

        conn = connect_to_db()
        if not conn:
            return jsonify({"status": "fail", "message": "Failed to connect to database"}), 500

        cursor = conn.execute(
            """
            SELECT 
                u.id, u.username, u.avatar, 
                (SELECT COUNT(*) FROM user_followers WHERE user_id = u.id) AS followers,
                (SELECT COUNT(*) FROM user_followers WHERE follower_id = u.id) AS following
            FROM user u
            LIMIT ? OFFSET ?
            """, (per_page, offset)
        )

        users = [
            {
                "id": row[0],
                "username": row[1],
                "avatar": row[2],
                "followers": row[3],
                "following": row[4]
            }
            for row in cursor.fetchall()
        ]
        conn.close()

        if users:
            return jsonify({"status": "success", "users": users})
        else:
            return jsonify({"status": "fail", "message": "No users found"}), 404

    except sqlite3.Error as e:
        return jsonify({"status": "fail", "message": f"Database error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"status": "fail", "message": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/toggle_follow', methods=['POST'])
def toggle_follow():
    try:
        data = request.json
        if not data or 'user_id' not in data or 'follower_id' not in data:
            return jsonify({"status": "fail", "message": "Missing required fields 'user_id' and 'follower_id'"}), 400

        user_id = data.get('user_id')
        follower_id = data.get('follower_id')

        if user_id == follower_id:
            return jsonify({"status": "fail", "message": "User cannot follow/unfollow themselves"}), 400

        conn = connect_to_db()
        if not conn:
            return jsonify({"status": "fail", "message": "Failed to connect to database"}), 500

        cursor = conn.execute("SELECT 1 FROM user WHERE id = ?", (user_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"status": "fail", "message": f"User with id {user_id} not found"}), 404

        cursor = conn.execute("SELECT 1 FROM user WHERE id = ?", (follower_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"status": "fail", "message": f"User with id {follower_id} not found"}), 404

        cursor = conn.execute(
            "SELECT 1 FROM user_followers WHERE user_id = ? AND follower_id = ?",
            (user_id, follower_id)
        )
        follow_exists = cursor.fetchone()

        if follow_exists:
            conn.execute(
                "DELETE FROM user_followers WHERE user_id = ? AND follower_id = ?",
                (user_id, follower_id)
            )
            conn.commit()
            conn.close()
            return jsonify({"status": "success", "message": "Follow removed successfully"})
        else:
            conn.execute(
                "INSERT INTO user_followers (user_id, follower_id) VALUES (?, ?)",
                (user_id, follower_id)
            )
            conn.commit()
            conn.close()
            return jsonify({"status": "success", "message": "Follow added successfully"})

    except sqlite3.Error as e:
        return jsonify({"status": "fail", "message": f"Database error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"status": "fail", "message": f"An unexpected error occurred: {str(e)}"}), 500
    
@app.route('/search_user', methods=['POST'])
def search_user():
    try:
        data = request.get_json()
        search_query = data.get('username', '').strip()
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)

        if page < 1 or per_page < 1:
            return jsonify({"status": "fail", "message": "Page and per_page must be greater than 0"}), 400

        search_query = unidecode(search_query.lower())
        search_query = search_query.strip()

        offset = (page - 1) * per_page

        conn = connect_to_db()
        if not conn:
            return jsonify({"status": "fail", "message": "Failed to connect to database"}), 500

        cursor = conn.execute(
            """
            SELECT 
                u.id, u.username, u.avatar, u.display_name, u.normal_display_name,
                (SELECT COUNT(*) FROM user_followers WHERE user_id = u.id) AS followers,
                (SELECT COUNT(*) FROM user_followers WHERE follower_id = u.id) AS following
            FROM user u
            WHERE u.normal_display_name LIKE ? 
            LIMIT ? OFFSET ?
            """, ('%' + search_query + '%', per_page, offset)
        )

        users = [
            {
                "id": row[0],
                "username": row[1],
                "avatar": row[2],
                "display_name": row[3],
                "normal_display_name": row[4],
                "followers": row[5],
                "following": row[6]
            }
            for row in cursor.fetchall()
        ]
        conn.close()

        if users:
            return jsonify({"status": "success", "users": users})
        else:
            return jsonify({"status": "fail", "message": "No users found"}), 404

    except sqlite3.Error as e:
        return jsonify({"status": "fail", "message": f"Database error: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"status": "fail", "message": f"An unexpected error occurred: {str(e)}"}), 500

# Hàm tạo đặc trưng cho các liên kết tiềm năng (potential links)
def generate_features_for_potential_links(data, feature_dict, model):
    combined_features = []
    for u, v in tqdm(data, desc="Preparing features for potential links"):
        if u in model.wv and v in model.wv:
            # Vector nhúng của u và v
            embedding_u = model.wv[u]
            embedding_v = model.wv[v]

            # Đặc trưng thủ công
            handcrafted_features = feature_dict.get((u, v), np.zeros(7))  # Nếu không có đặc trưng, thay bằng 0

            # Kết hợp vector nhúng và đặc trưng thủ công
            combined_features.append(
                np.concatenate([embedding_u, embedding_v, handcrafted_features])
            )
        else:
            combined_features.append(None)  # Lưu None nếu không có vector nhúng
    return combined_features

def recommend_links_for_node(input_node, graph, model, node2vec_model, nodes_features, top_k=10):
    # Lấy tất cả các node trong đồ thị
    all_nodes = list(graph.nodes())

    # Tạo danh sách các cặp node tiềm năng (u, v) với u là input_node và v là tất cả các node khác
    potential_links = [(input_node, v) for v in all_nodes if v != input_node and not graph.has_edge(input_node, v)]

    # Tạo đặc trưng cho các potential links
    X_potential_links = generate_features_for_potential_links(potential_links, nodes_features, node2vec_model)

    # Dự đoán xác suất cho các liên kết tiềm năng
    predicted_probs = model.predict_proba(X_potential_links)[:, 1]  # Lấy xác suất của lớp 1 (có liên kết)

    # Gợi ý top_k liên kết có xác suất cao nhất
    top_k_indices = np.argsort(predicted_probs)[-top_k:][::-1]
    top_k_recommendations = [potential_links[i] for i in top_k_indices]
    top_k_scores = [predicted_probs[i] for i in top_k_indices]

    return top_k_recommendations, top_k_scores

# Đọc đồ thị và tải mô hình
G = nx.read_graphml(processed_graph_path)
clf = joblib.load(model_checkpoint)
with open(node2vec_full_model_checkpoint_path, 'rb') as f:
    node2vec_model = pickle.load(f)
    
with open(nodes_features_checkpoint_path, 'rb') as f:
    nodes_features = pickle.load(f)

@app.route('/recommend', methods=['GET'])
def recommend():
    input_node = request.args.get('input_node')
    top_k = int(request.args.get('top_k', 10))

    if not input_node:
        return jsonify({"error": "input_node is required"}), 400

    try:
        # Gợi ý các liên kết tiềm năng và điểm số
        recommendations, scores = recommend_links_for_node(input_node, G, clf, node2vec_model, nodes_features, top_k)
        print(recommendations)

        # Kết nối đến cơ sở dữ liệu
        conn = connect_to_db()
        if not conn:
            return jsonify({"error": "Failed to connect to database"}), 500

        response = []
        for link, score in zip(recommendations, scores):
            user_id = link[0]
            user_id_predict = link[1]

            cursor = conn.execute(
                """
                SELECT 
                    u.id, u.username, u.avatar, u.display_name,
                    (SELECT COUNT(*) FROM user_followers WHERE user_id = u.id) AS followers,
                    (SELECT COUNT(*) FROM user_followers WHERE follower_id = u.id) AS following
                FROM user u WHERE u.id = ?
                """,
                (user_id_predict,)
            )
            user_predict = cursor.fetchone()

            if user_predict:
                response.append({
                    'user_id': user_id,
                    'user_predict': {
                        "id": user_predict[0],
                        "username": user_predict[1],
                        "avatar": user_predict[2],
                        "display_name": user_predict[3],
                        "followers": user_predict[4],   
                        "following": user_predict[5]
                    },
                    'score': score
                })

        conn.close()

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)