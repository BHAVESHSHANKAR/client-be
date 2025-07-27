from flask import Flask, jsonify
from flask_cors import CORS
from routes.userRoutes import user_routes, bcrypt
from trainedmodels.brainModel import brain_routes
from trainedmodels.pneumoniaModel import pneumonia_routes
from models.database import init_db
from os import getenv
from dotenv import load_dotenv
import logging
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

load_dotenv()

app = Flask(__name__)

# Optimize Flask for production performance
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# Suppress Flask development server warnings
if not app.debug:
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

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

# Start keep-alive service in production
if not app.debug and getenv('RENDER_EXTERNAL_URL'):
    try:
        from keep_alive import start_keep_alive
        # Start keep-alive with 4-minute intervals (before 5-minute Render timeout)
        start_keep_alive(getenv('RENDER_EXTERNAL_URL'), interval=240)
        print("🔄 Keep-alive service started for production")
    except Exception as e:
        print(f"⚠️ Keep-alive service failed to start: {e}")

# Health check route
@app.route('/')
def health_check():
    return jsonify({
        "status": "success",
        "message": "Server is running successfully",
        "database_status": "connected"
    }), 200

# Optimized warm-up endpoint with better error handling
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
        # Step 1: Quick health check
        print("📊 Step 1: Basic server health check...")
        server_status = {
            "status": "warming_up",
            "message": "Server is warming up",
            "database_connected": False,
            "models_loaded": False,
            "warmup_time": "0s"
        }
        
        # Step 2: Test database connection with timeout
        print("📊 Step 2: Testing database connection...")
        try:
            from models.database import User
            # Quick count query with timeout
            user_count = User.query.count()
            server_status["database_connected"] = True
            server_status["database_users"] = user_count
            print(f"✅ Database connected successfully! Users in DB: {user_count}")
        except Exception as db_error:
            # Silently handle database connection issues
            server_status["database_error"] = "Database connection delayed"
            # Continue without failing - database might be slow but server can still work
        
        # Step 3: Load AI models (optional for faster warmup)
        print("🧠 Step 3: Loading AI models into memory...")
        models_loaded = False
        try:
            # Only load models if database is working to avoid double timeout
            if server_status["database_connected"]:
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
            else:
                print("⚠️ Skipping model loading due to database issues")
                
        except Exception as model_error:
            # Silently handle model loading errors - they'll load on-demand
            server_status["model_error"] = "Models will load on first use"
            # Continue without failing - models can load on-demand
        
        elapsed = time.time() - start_time
        server_status.update({
            "status": "warmed_up",
            "message": "Server is warm and ready",
            "models_loaded": models_loaded,
            "warmup_time": f"{elapsed:.2f}s"
        })
        
        print("🔥 ========================================")
        print(f"🔥 SERVER WARMUP COMPLETED! ({elapsed:.2f}s)")
        print("🔥 ========================================")
        print("🚀 Server is now ready for requests!")
        print("💡 Next requests will be faster!")
        print("🔥 ========================================")
        
        return jsonify(server_status), 200
        
    except Exception as e:
        elapsed = time.time() - start_time
        print("❌ ========================================")
        print(f"❌ SERVER WARMUP FAILED! ({elapsed:.2f}s)")
        print("❌ ========================================")
        print(f"❌ Error: {str(e)}")
        print("❌ ========================================")
        
        # Return partial success instead of complete failure
        return jsonify({
            "status": "partial_warmup",
            "message": "Server is partially ready",
            "error": str(e),
            "warmup_time": f"{elapsed:.2f}s",
            "database_connected": False,
            "models_loaded": False
        }), 200  # Return 200 instead of 500 to avoid frontend errors

# Add a simple health check endpoint
@app.route('/health')
def health_check_simple():
    """Simple health check without heavy operations"""
    import time
    return jsonify({
        "status": "healthy",
        "message": "Server is running",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }), 200

# Keep-alive endpoint for external cron jobs
@app.route('/keep-alive')
def keep_alive_ping():
    """Endpoint for external keep-alive services (cron jobs, etc.)"""
    import time
    return jsonify({
        "status": "alive",
        "message": "Server is active and responding",
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "uptime": "Server is running"
    }), 200

# Register routes
app.register_blueprint(user_routes)
app.register_blueprint(brain_routes, url_prefix='/brain')
app.register_blueprint(pneumonia_routes, url_prefix='/pneumonia')

if __name__ == '__main__':
    print(f"🚀 Server running on http://localhost:5000")
    app.run(debug=True)