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
    print("✅ Neon PostgreSQL connection successfully")
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

# Warm-up endpoint to keep server active
@app.route('/warmup')
def warmup():
    """Endpoint to warm up the server and prevent cold starts"""
    import time
    start_time = time.time()
    
    print("🔥 ========================================")
    print("🔥 SERVER WARMUP INITIATED!")
    print("🔥 ========================================")
    print(f"🕐 Warmup started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        print("📊 Step 1: Testing database connection...")
        # Test database connection
        from models.database import User
        user_count = User.query.count()
        print(f"✅ Database connected successfully! Users in DB: {user_count}")
        
        print("🧠 Step 2: Loading AI models into memory...")
        # Warm up models (optional - loads them into memory)
        # This ensures first analysis is faster
        try:
            from trainedmodels.brainModel import get_model
            from trainedmodels.pneumoniaModel import get_pneumonia_model
            
            print("🧠 Loading brain tumor detection model...")
            brain_model, brain_metrics = get_model()
            print(f"✅ Brain model loaded! Accuracy: {brain_metrics['accuracy']}")
            
            print("🫁 Loading pneumonia detection model...")
            pneumonia_model = get_pneumonia_model()
            print("✅ Pneumonia model loaded successfully!")
            
            models_loaded = True
            print("🎯 All AI models are now in memory and ready!")
            
        except Exception as e:
            print(f"⚠️ Models not pre-loaded: {e}")
            models_loaded = False
        
        elapsed = time.time() - start_time
        
        print("🔥 ========================================")
        print(f"🔥 SERVER WARMUP COMPLETED! ({elapsed:.2f}s)")
        print("🔥 ========================================")
        print("🚀 Server is now ready for lightning-fast requests!")
        print("💡 Next requests will be super fast!")
        print("🔥 ========================================")
        
        return jsonify({
            "status": "warmed_up",
            "message": "Server is warm and ready",
            "database_users": user_count,
            "models_loaded": models_loaded,
            "warmup_time": f"{elapsed:.2f}s"
        }), 200
        
    except Exception as e:
        elapsed = time.time() - start_time
        print("❌ ========================================")
        print(f"❌ SERVER WARMUP FAILED! ({elapsed:.2f}s)")
        print("❌ ========================================")
        print(f"❌ Error: {str(e)}")
        print("❌ ========================================")
        
        return jsonify({
            "status": "error",
            "message": f"Warmup failed: {str(e)}"
        }), 500

# Register routes
app.register_blueprint(user_routes)
app.register_blueprint(brain_routes, url_prefix='/brain')
app.register_blueprint(pneumonia_routes, url_prefix='/pneumonia')

if __name__ == '__main__':
    print(f"🚀 Server running on http://localhost:5000")
    app.run(debug=True)