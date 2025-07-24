from flask import Blueprint, jsonify, request
import numpy as np
import cv2
import tensorflow as tf
import base64
import os
import jwt
import time
from functools import wraps
from os import getenv
from dotenv import load_dotenv
from models.database import db, BrainAnalysis
from utils.cloudinary_client import cloudinary_client

load_dotenv()
SECRET_KEY = getenv("SECRET_KEY")

brain_routes = Blueprint('brain_routes', __name__)

# JWT Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        print("🔐 Authentication check started")  # Debug
        token = None
        
        # Check for token in Authorization header
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            print(f"📝 Authorization header found: {auth_header[:20]}...")  # Debug (partial)
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
                print("✅ Token extracted successfully")  # Debug
            except IndexError:
                print("❌ Invalid token format")  # Debug
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid token format. Use: Bearer <token>'
                }), 401
        else:
            print("❌ No Authorization header found")  # Debug
        
        if not token:
            print("❌ Token is missing")  # Debug
            return jsonify({
                'status': 'error',
                'message': 'Token is missing. Please login first.'
            }), 401
        
        if not SECRET_KEY:
            print("❌ SECRET_KEY is missing from environment")  # Debug
            return jsonify({
                'status': 'error',
                'message': 'Server configuration error'
            }), 500
        
        try:
            # Decode the token
            print("� Attemepting to decode token...")  # Debug
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user_id = data['user_id']
            current_username = data['username']
            print(f"✅ Token decoded successfully for user: {current_username}")  # Debug
        except jwt.ExpiredSignatureError:
            print("❌ Token has expired")  # Debug
            return jsonify({
                'status': 'error',
                'message': 'Token has expired. Please login again.'
            }), 401
        except jwt.InvalidTokenError as e:
            print(f"❌ Invalid token: {e}")  # Debug
            return jsonify({
                'status': 'error',
                'message': 'Invalid token. Please login again.'
            }), 401
        
        # Pass user info to the route
        print("🎯 Authentication successful, proceeding to route")  # Debug
        return f(current_user_id, current_username, *args, **kwargs)
    
    return decorated

# Global variables for lazy loading
_model = None
_metrics = None

# Load model and metrics (lazy loading)
def get_model():
    global _model, _metrics
    
    if _model is None:
        # Use a relative path from the location of brainModel.py
        model_path = os.path.join(os.path.dirname(__file__), 'brain_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        print("🧠 Loading brain model...")
        _model = tf.keras.models.load_model(model_path)
        print("✅ Brain Model loaded successfully")
        
        _metrics = {
            'accuracy': 0.92,
            'loss': 0.15,
            'precision': 0.91,
            'recall': 0.90,
            'auc': 0.94
        }
    
    return _model, _metrics

# Utility Functions
def is_likely_mri(image):
    try:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if min(gray.shape) < 32:
            return False

        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            if np.mean(hsv[:, :, 1]) > 100:
                return False

        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        if np.std(normalized) < 10:
            return False

        return True
    except:
        return False

def check_brain_structure(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[-1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    ratio = (w * h) / (image.shape[0] * image.shape[1])
    if ratio < 0.1 or ratio > 0.95:
        return False

    ar = w / h
    if ar < 0.5 or ar > 2:
        return False

    return True

def crop_img(gray, color):
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return color[y:y+h, x:x+w]

def encode_image(img):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# MRI analysis route - Protected with JWT authentication
@brain_routes.route('/analyze', methods=['POST'])
@token_required
def analyze_mri(current_user_id, current_username):
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded."}), 400

    file = request.files['file']
    try:
        img_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128, 128))
        image_base64 = encode_image(image)

        if not is_likely_mri(image):
            return jsonify({
                "status": "error",
                "message": "Not a valid MRI scan. Upload grayscale/low-saturation brain MRI."
            }), 400

        if not check_brain_structure(image):
            return jsonify({
                "status": "error",
                "message": "Brain structure not detected. Upload a centered, clear brain scan."
            }), 400

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cropped = crop_img(gray, image)

        if cropped is None or cropped.size == 0:
            return jsonify({
                "status": "error",
                "message": "Cropping failed. Try a clearer scan."
            }), 400

        # Load model only when needed
        model, metrics = get_model()
        
        cropped_resized = cv2.resize(cropped, (50, 50))
        prediction = model.predict(np.expand_dims(cropped_resized, axis=0))
        confidence = float(prediction[0][0])

        # Record processing time
        start_time = time.time()
        
        # Generate processed images for upload
        cropped_display = cv2.resize(cropped, (128, 128))
        gray_mask = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray_mask, 0, 255, cv2.THRESH_OTSU)[-1]
        thresh_display = cv2.resize(thresh, (128, 128))
        
        # Upload images to Cloudinary
        print("📤 Uploading images to Cloudinary...")
        image_urls = cloudinary_client.upload_brain_analysis_images(
            original_image=image,
            cropped_image=cropped_display,
            threshold_image=thresh_display,
            user_id=current_user_id
        )
        
        if confidence >= 0.5:
            prediction_result = {
                "tumor_detected": True,
                "confidence": confidence * 100,
                "message": f"Tumor detected with confidence: {confidence * 100:.2f}%"
            }
        else:
            prediction_result = {
                "tumor_detected": False,
                "confidence": (1 - confidence) * 100,
                "message": f"No tumor detected with confidence: {(1 - confidence) * 100:.2f}%"
            }

        processing_time = time.time() - start_time

        # Save analysis to database with Cloudinary image URLs
        try:
            analysis = BrainAnalysis(
                user_id=current_user_id,
                tumor_detected=prediction_result["tumor_detected"],
                confidence=prediction_result["confidence"],
                model_accuracy=metrics['accuracy'],
                model_loss=metrics['loss'],
                model_precision=metrics['precision'],
                model_recall=metrics['recall'],
                model_auc=metrics['auc'],
                image_filename=file.filename if file.filename else 'unknown.jpg',
                image_size="128x128",
                processing_time=processing_time,
                original_image_url=image_urls.get('original_image_url'),
                cropped_image_url=image_urls.get('cropped_image_url'),
                threshold_image_url=image_urls.get('threshold_image_url')
            )
            
            db.session.add(analysis)
            db.session.commit()
            print(f"✅ Analysis saved to database with ID: {analysis.id}")
            
        except Exception as db_error:
            print(f"❌ Failed to save analysis to database: {db_error}")
            db.session.rollback()
            # Continue without failing the request

        result = {
            "status": "success",
            "prediction": prediction_result,
            "metrics": metrics,
            "analysis_id": analysis.id if 'analysis' in locals() else None
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Processing failed: {str(e)}"
        }), 500

# Get user's brain analysis history
@brain_routes.route('/history', methods=['GET'])
@token_required
def get_analysis_history(current_user_id, current_username):
    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Query user's analyses with pagination
        analyses = BrainAnalysis.query.filter_by(user_id=current_user_id)\
                                    .order_by(BrainAnalysis.created_at.desc())\
                                    .paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            "status": "success",
            "analyses": [analysis.to_dict() for analysis in analyses.items],
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": analyses.total,
                "pages": analyses.pages,
                "has_next": analyses.has_next,
                "has_prev": analyses.has_prev
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to get analysis history: {str(e)}"
        }), 500

# Get specific analysis by ID
@brain_routes.route('/analysis/<int:analysis_id>', methods=['GET'])
@token_required
def get_analysis(current_user_id, current_username, analysis_id):
    try:
        analysis = BrainAnalysis.query.filter_by(
            id=analysis_id, 
            user_id=current_user_id
        ).first()
        
        if not analysis:
            return jsonify({
                "status": "error",
                "message": "Analysis not found"
            }), 404
        
        return jsonify({
            "status": "success",
            "analysis": analysis.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to get analysis: {str(e)}"
        }), 500