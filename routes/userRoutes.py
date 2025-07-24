from flask import Blueprint, request, jsonify
from flask_bcrypt import Bcrypt
from models.database import db, User
import jwt
import datetime
from os import getenv, environ
from dotenv import load_dotenv
from sqlalchemy.exc import IntegrityError

load_dotenv()

# Configuration
try:
    SECRET_KEY = environ["SECRET_KEY"]
except KeyError:
    raise ValueError("SECRET_KEY environment variable must be set")

bcrypt = Bcrypt()
user_routes = Blueprint('user_routes', __name__)

# ------------------- REGISTER -------------------
@user_routes.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get("username")
        email = data.get("email")
        password = data.get("password")
        age = data.get("age")

        # Validation
        if not all([username, email, password, age]):
            return jsonify({"error": "All fields are required"}), 400

        # Check if user already exists
        existing_user = User.query.filter(
            (User.email == email) | (User.username == username)
        ).first()
        
        if existing_user:
            if existing_user.email == email:
                return jsonify({"error": "Email already exists"}), 400
            else:
                return jsonify({"error": "Username already exists"}), 400

        # Create new user
        new_user = User(
            username=username,
            email=email,
            age=int(age)
        )
        new_user.set_password(password)

        # Save to database
        db.session.add(new_user)
        db.session.commit()

        return jsonify({
            "message": "User registered successfully",
            "user": new_user.to_dict()
        }), 201

    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "Username or email already exists"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid data: {str(e)}"}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

# ------------------- LOGIN -------------------
@user_routes.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        # Find user by email
        user = User.query.filter_by(email=email).first()
        if not user:
            return jsonify({"error": "Invalid email or password"}), 401

        # Check password
        if not user.check_password(password):
            return jsonify({"error": "Invalid email or password"}), 401

        # Create JWT token
        payload = {
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
        }

        token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

        return jsonify({
            "message": "Login successful",
            "token": token,
            "user": user.to_dict()
        }), 200

    except Exception as e:
        return jsonify({"error": f"Login failed: {str(e)}"}), 500

# ------------------- GET USER PROFILE -------------------
@user_routes.route('/profile/<int:user_id>', methods=['GET'])
def get_profile(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "status": "success",
            "user": user.to_dict()
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to get profile: {str(e)}"}), 500

# ------------------- UPDATE USER PROFILE -------------------
@user_routes.route('/profile/<int:user_id>', methods=['PUT'])
def update_profile(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        data = request.get_json()
        
        # Update allowed fields
        if 'username' in data:
            # Check if username is already taken by another user
            existing_user = User.query.filter(
                User.username == data['username'],
                User.id != user_id
            ).first()
            if existing_user:
                return jsonify({"error": "Username already exists"}), 400
            user.username = data['username']
        
        if 'age' in data:
            user.age = int(data['age'])
        
        if 'password' in data:
            user.set_password(data['password'])

        db.session.commit()

        return jsonify({
            "message": "Profile updated successfully",
            "user": user.to_dict()
        }), 200

    except IntegrityError:
        db.session.rollback()
        return jsonify({"error": "Username already exists"}), 400
    except ValueError as e:
        db.session.rollback()
        return jsonify({"error": f"Invalid data: {str(e)}"}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to update profile: {str(e)}"}), 500

# ------------------- DELETE USER -------------------
@user_routes.route('/profile/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        db.session.delete(user)
        db.session.commit()

        return jsonify({"message": "User deleted successfully"}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Failed to delete user: {str(e)}"}), 500

# ------------------- GET ALL USERS (Admin) -------------------
@user_routes.route('/users', methods=['GET'])
def get_all_users():
    try:
        users = User.query.all()
        return jsonify({
            "status": "success",
            "users": [user.to_dict() for user in users],
            "total": len(users)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to get users: {str(e)}"}), 500