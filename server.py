from flask import Flask, jsonify
from flask_cors import CORS
from routes.userRoutes import user_routes, bcrypt
from trainedmodels.brainModel import brain_routes
from trainedmodels.pneumoniaModel import pneumonia_routes
from models.database import init_db
from os import getenv
from dotenv import load_dotenv
import logging

load_dotenv()

app = Flask(__name__)

# Optimize Flask for production performance
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Database configuration with optimizations
DATABASE_URL = getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable must be set")

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Database connection pooling for faster queries
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_size': 10,
    'pool_recycle': 120,
    'pool_pre_ping': True,
    'max_overflow': 20,
    'pool_timeout': 30
}

# CORS setup
FRONTEND_URL = getenv("FRONTEND_URL")
CORS(app, resources={
    r"/*": {
        "origins": FRONTEND_URL,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize extensions
bcrypt.init_app(app)

# Initialize database
try:
    init_db(app)
    print("✅ Neon PostgreSQL connection successful")
except Exception as e:
    print("❌ Database connection failed:", e)
    raise e

# Health check route
@app.route('/')
def health_check():
    return jsonify({
        "status": "success",
        "message": "Server is running successfully",
        "database_status": "connected"
    }), 200

# Register routes
app.register_blueprint(user_routes)
app.register_blueprint(brain_routes, url_prefix='/brain')
app.register_blueprint(pneumonia_routes, url_prefix='/pneumonia')

if __name__ == '__main__':
    print(f"🚀 Server running on http://localhost:5000")
    app.run(debug=True)