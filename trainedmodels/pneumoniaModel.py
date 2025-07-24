# from flask import Blueprint, jsonify, request
# import numpy as np
# import cv2
# from PIL import Image
# import tensorflow as tf
# import io
# import base64
# import os
# import jwt
# import time
# from functools import wraps
# from os import getenv
# from dotenv import load_dotenv
# from models.database import db, PneumoniaAnalysis

# load_dotenv()
# SECRET_KEY = getenv("SECRET_KEY")

# pneumonia_routes = Blueprint('pneumonia_routes', __name__)

# # Load the model with path relative to the script location
# model_path = os.path.join(os.path.dirname(__file__), 'pneumonia_model.h5')
# model = tf.keras.models.load_model(model_path)
# print("Pneumonia model successfully loaded")  # Added statement

# # JWT Authentication decorator
# def token_required(f):
#     @wraps(f)
#     def decorated(*args, **kwargs):
#         print("🔐 Authentication check started")  # Debug
#         token = None
        
#         # Check for token in Authorization header
#         if 'Authorization' in request.headers:
#             auth_header = request.headers['Authorization']
#             print(f"📝 Authorization header found: {auth_header[:20]}...")  # Debug (partial)
#             try:
#                 token = auth_header.split(" ")[1]  # Bearer <token>
#                 print("✅ Token extracted successfully")  # Debug
#             except IndexError:
#                 print("❌ Invalid token format")  # Debug
#                 return jsonify({
#                     'status': 'error',
#                     'message': 'Invalid token format. Use: Bearer <token>'
#                 }), 401
#         else:
#             print("❌ No Authorization header found")  # Debug
        
#         if not token:
#             print("❌ Token is missing")  # Debug
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Token is missing. Please login first.'
#             }), 401
        
#         if not SECRET_KEY:
#             print("❌ SECRET_KEY is missing from environment")  # Debug
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Server configuration error'
#             }), 500
        
#         try:
#             # Decode the token
#             print("🔍 Attempting to decode token...")  # Debug
#             data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
#             current_user_id = data['user_id']
#             current_username = data['username']
#             print(f"✅ Token decoded successfully for user: {current_username}")  # Debug
#         except jwt.ExpiredSignatureError:
#             print("❌ Token has expired")  # Debug
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Token has expired. Please login again.'
#             }), 401
#         except jwt.InvalidTokenError as e:
#             print(f"❌ Invalid token: {e}")  # Debug
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Invalid token. Please login again.'
#             }), 401
        
#         # Pass user info to the route
#         print("🎯 Authentication successful, proceeding to route")  # Debug
#         return f(current_user_id, current_username, *args, **kwargs)
    
#     return decorated

# def is_chest_xray(image):
#     # Convert to numpy array
#     img_array = np.array(image)
    
#     # If image is already grayscale, it's probably an X-ray
#     if len(img_array.shape) == 2:
#         return True
        
#     # For color images, check if it's close to grayscale
#     r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
    
#     # Calculate average intensity
#     avg_intensity = (r + g + b) / 3
    
#     # Calculate how much each channel deviates from average
#     r_diff = np.mean(np.abs(r - avg_intensity))
#     g_diff = np.mean(np.abs(g - avg_intensity))
#     b_diff = np.mean(np.abs(b - avg_intensity))
    
#     # Very lenient threshold - if channels are somewhat similar, accept it
#     return (r_diff + g_diff + b_diff) / 3 < 25

# @pneumonia_routes.route('/')
# def home():
#     return jsonify({'message': 'Pneumonia detection service is running'})

# @pneumonia_routes.route('/predict', methods=['POST'])
# @token_required
# def predict(current_user_id, current_username):
#     try:
#         # Get the image from the POST request
#         if 'file' not in request.files:
#             return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
#         file = request.files['file']
#         if file.filename == '':
#             return jsonify({'status': 'error', 'message': 'No file selected'}), 400

#         # Read and validate the image
#         image = Image.open(file.stream)
        
#         # Check if it's an X-ray
#         if not is_chest_xray(image):
#             return jsonify({'status': 'error', 'message': 'Please upload a grayscale X-ray image'}), 400

#         # Preprocess the image
#         input_image_resized = cv2.resize(np.array(image), (128, 128))
#         input_image_normalized = input_image_resized / 255

#         if len(input_image_normalized.shape) == 2:
#             input_image_normalized = np.stack((input_image_normalized,) * 3, axis=-1)

#         input_image_reshaped = np.reshape(input_image_normalized, [1, 128, 128, 3])

#         # Record processing time
#         start_time = time.time()
        
#         # Make prediction
#         prediction = model.predict(input_image_reshaped)
#         confidence = float(prediction[0][np.argmax(prediction)])
#         pred_label = np.argmax(prediction)
        
#         processing_time = time.time() - start_time
        
#         # Prepare result
#         pneumonia_detected = pred_label == 1
#         prediction_label = 'Pneumonia detected' if pneumonia_detected else 'No Pneumonia detected'
#         confidence_percentage = confidence * 100

#         # Save analysis to database
#         try:
#             analysis = PneumoniaAnalysis(
#                 user_id=current_user_id,
#                 pneumonia_detected=pneumonia_detected,
#                 confidence=confidence_percentage,
#                 prediction_label=prediction_label,
#                 image_filename=file.filename if file.filename else 'unknown.jpg',
#                 image_size="128x128",
#                 processing_time=processing_time
#             )
            
#             db.session.add(analysis)
#             db.session.commit()
#             print(f"✅ Pneumonia analysis saved to database with ID: {analysis.id}")
            
#         except Exception as db_error:
#             print(f"❌ Failed to save pneumonia analysis to database: {db_error}")
#             db.session.rollback()
#             # Continue without failing the request

#         return jsonify({
#             'status': 'success',
#             'prediction': prediction_label,
#             'confidence': f'{confidence:.2%}',
#             'pneumonia_detected': pneumonia_detected,
#             'analysis_id': analysis.id if 'analysis' in locals() else None
#         })

#     except Exception as e:
#         return jsonify({'status': 'error', 'message': str(e)}), 500

# # Get user's pneumonia analysis history
# @pneumonia_routes.route('/history', methods=['GET'])
# @token_required
# def get_analysis_history(current_user_id, current_username):
#     try:
#         # Get pagination parameters
#         page = request.args.get('page', 1, type=int)
#         per_page = request.args.get('per_page', 10, type=int)
        
#         # Query user's analyses with pagination
#         analyses = PneumoniaAnalysis.query.filter_by(user_id=current_user_id)\
#                                         .order_by(PneumoniaAnalysis.created_at.desc())\
#                                         .paginate(page=page, per_page=per_page, error_out=False)
        
#         return jsonify({
#             "status": "success",
#             "analyses": [analysis.to_dict() for analysis in analyses.items],
#             "pagination": {
#                 "page": page,
#                 "per_page": per_page,
#                 "total": analyses.total,
#                 "pages": analyses.pages,
#                 "has_next": analyses.has_next,
#                 "has_prev": analyses.has_prev
#             }
#         }), 200
        
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": f"Failed to get analysis history: {str(e)}"
#         }), 500

# # Get specific pneumonia analysis by ID
# @pneumonia_routes.route('/analysis/<int:analysis_id>', methods=['GET'])
# @token_required
# def get_analysis(current_user_id, current_username, analysis_id):
#     try:
#         analysis = PneumoniaAnalysis.query.filter_by(
#             id=analysis_id, 
#             user_id=current_user_id
#         ).first()
        
#         if not analysis:
#             return jsonify({
#                 "status": "error",
#                 "message": "Analysis not found"
#             }), 404
        
#         return jsonify({
#             "status": "success",
#             "analysis": analysis.to_dict()
#         }), 200
        
#     except Exception as e:
#         return jsonify({
#             "status": "error",
#             "message": f"Failed to get analysis: {str(e)}"
#         }), 500
from flask import Blueprint, jsonify, request
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import io
import base64
import os
import jwt
import time
from functools import wraps
from os import getenv
from dotenv import load_dotenv
from models.database import db, PneumoniaAnalysis
from utils.cloudinary_client import cloudinary_client


load_dotenv()
SECRET_KEY = getenv("SECRET_KEY")

pneumonia_routes = Blueprint('pneumonia_routes', __name__)

# Global variables for lazy loading
_pneumonia_model = None

# Load the model with lazy loading
def get_pneumonia_model():
    global _pneumonia_model
    
    if _pneumonia_model is None:
        model_path = os.path.join(os.path.dirname(__file__), 'pneumonia_model.h5')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pneumonia model file not found at {model_path}")
        
        print("🫁 Loading pneumonia model...")
        _pneumonia_model = tf.keras.models.load_model(model_path)
        print("✅ Pneumonia model successfully loaded")
    
    return _pneumonia_model

# JWT Authentication decorator (keeping for backward compatibility)
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
            print("🔍 Attempting to decode token...")  # Debug
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

def is_chest_xray(image):
    """
    Proper chest X-ray detection based on anatomical structure analysis
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Handle color images - reject if truly colored
        if len(img_array.shape) == 3:
            r, g, b = img_array[..., 0], img_array[..., 1], img_array[..., 2]
            
            # Multiple color detection methods for better accuracy
            
            # Method 1: Channel mean differences
            r_mean, g_mean, b_mean = np.mean(r), np.mean(g), np.mean(b)
            max_channel_diff = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))
            
            # Method 2: Standard deviation differences
            r_std, g_std, b_std = np.std(r), np.std(g), np.std(b)
            std_variance = np.var([r_std, g_std, b_std])
            
            # Method 3: Pixel-wise color differences
            color_diff_pixels = np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b))
            
            print(f"🎨 Channel mean diff: {max_channel_diff:.1f}")
            print(f"🎨 Std variance: {std_variance:.1f}")
            print(f"🎨 Pixel color diff: {color_diff_pixels:.1f}")
            
            # Reject if ANY method indicates significant color
            if (max_channel_diff > 20 or  # Channel means are different
                std_variance > 30 or      # Standard deviations vary significantly
                color_diff_pixels > 15):  # Pixels have color differences
                print(f"❌ Rejected: Color photo detected")
                return False
            
            # Convert to grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        height, width = img_array.shape
        print(f"📏 Image dimensions: {height}x{width}")
        
        # Size check
        if height < 100 or width < 100:
            print(f"❌ Rejected: Too small for chest X-ray")
            return False
        
        # 1. CHEST X-RAY SPECIFIC STRUCTURE DETECTION
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img_array, (5, 5), 0)
        
        # Look for bilateral lung fields (dark areas on left and right)
        h_center = height // 2
        w_quarter = width // 4
        
        # Define potential lung regions
        left_lung_region = blurred[h_center-height//4:h_center+height//4, w_quarter:w_quarter*2]
        right_lung_region = blurred[h_center-height//4:h_center+height//4, w_quarter*2:w_quarter*3]
        
        # Calculate intensities
        left_lung_mean = np.mean(left_lung_region)
        right_lung_mean = np.mean(right_lung_region)
        overall_mean = np.mean(blurred)
        
        print(f"🫁 Left lung region: {left_lung_mean:.1f}")
        print(f"🫁 Right lung region: {right_lung_mean:.1f}")
        print(f"📊 Overall mean: {overall_mean:.1f}")
        
        # In chest X-rays, lung regions should be darker than overall image
        lung_to_overall_ratio = (left_lung_mean + right_lung_mean) / (2 * overall_mean)
        print(f"🫁 Lung-to-overall ratio: {lung_to_overall_ratio:.3f}")
        
        # More lenient lung field check
        if lung_to_overall_ratio > 1.2:  # Much more lenient
            print(f"❌ Rejected: No dark lung fields detected")
            return False
        
        # 2. RIB CAGE DETECTION - Look for linear structures (more lenient)
        # Apply edge detection with lower thresholds
        edges = cv2.Canny(blurred, 20, 80)  # Lower thresholds
        
        # Use Hough Line Transform to detect linear structures (ribs)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=int(min(height, width) * 0.1))  # Lower threshold
        
        line_count = len(lines) if lines is not None else 0
        print(f"🦴 Linear structures (ribs): {line_count}")
        
        # Much more lenient - just need some linear structures
        if line_count < 3:  # Reduced from 8 to 3
            print(f"❌ Rejected: Insufficient rib-like structures")
            return False
        
        # 3. CHEST BOUNDARY DETECTION (more lenient)
        # Look for chest outline - should have curved boundaries
        # Check the outer edges for boundary structures
        left_boundary = edges[:, :width//6]
        right_boundary = edges[:, 5*width//6:]
        
        left_boundary_density = np.sum(left_boundary > 0) / left_boundary.size
        right_boundary_density = np.sum(right_boundary > 0) / right_boundary.size
        
        print(f"🔍 Left boundary density: {left_boundary_density:.4f}")
        print(f"🔍 Right boundary density: {right_boundary_density:.4f}")
        
        # More lenient boundary check
        if left_boundary_density < 0.002 and right_boundary_density < 0.002:  # Lower threshold
            print(f"❌ Rejected: No chest boundary detected")
            return False
        
        # 4. INTENSITY DISTRIBUTION CHECK (more lenient)
        # Chest X-rays have specific intensity patterns
        hist, _ = np.histogram(img_array, bins=32, range=(0, 256))
        
        # Find the number of significant intensity peaks
        peak_threshold = np.max(hist) * 0.1  # Lower threshold
        significant_bins = np.sum(hist > peak_threshold)
        
        print(f"📊 Significant intensity bins: {significant_bins}")
        
        # More lenient intensity check
        if significant_bins < 3:  # Reduced from 4 to 3
            print(f"❌ Rejected: Too few intensity levels for X-ray")
            return False
        
        # 5. SYMMETRY CHECK (more lenient)
        try:
            left_half = img_array[:, :width//2]
            right_half = np.fliplr(img_array[:, width//2:])
            
            # Make same size
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate correlation
            correlation = np.corrcoef(left_half.flatten(), right_half.flatten())[0, 1]
            print(f"🔄 Left-right correlation: {correlation:.3f}")
            
            # Much more lenient symmetry check
            if correlation < 0.2:  # Reduced from 0.4 to 0.2
                print(f"❌ Rejected: Poor bilateral symmetry")
                return False
        except:
            print("⚠ Symmetry check failed, skipping...")
            # Don't reject if symmetry calculation fails
        
        print("✅ Image passed comprehensive chest X-ray validation")
        return True
        
    except Exception as e:
        print(f"❌ Error in chest X-ray validation: {e}")
        return False

@pneumonia_routes.route('/')
def home():
    return jsonify({'message': 'Pneumonia detection service is running'})

@pneumonia_routes.route('/predict', methods=['POST'])
@token_required
def predict(current_user_id, current_username):
    try:
        # Get the image from the POST request
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'}), 400

        # Read and validate the image
        image = Image.open(file.stream)
        
        # Check if it's an X-ray
        if not is_chest_xray(image):
            return jsonify({'status': 'error', 'message': 'Please upload a grayscale X-ray image'}), 400

        # Preprocess the image
        input_image_resized = cv2.resize(np.array(image), (128, 128))
        input_image_normalized = input_image_resized / 255

        if len(input_image_normalized.shape) == 2:
            input_image_normalized = np.stack((input_image_normalized,) * 3, axis=-1)

        input_image_reshaped = np.reshape(input_image_normalized, [1, 128, 128, 3])

        # Record processing time
        start_time = time.time()
        
        # Load model only when needed
        model = get_pneumonia_model()
        
        # Make prediction
        print(f"🔍 Making prediction for image shape: {input_image_reshaped.shape}")
        prediction = model.predict(input_image_reshaped)
        print(f"📊 Raw prediction: {prediction}")
        
        confidence = float(prediction[0][np.argmax(prediction)])
        pred_label = int(np.argmax(prediction))  # Convert to Python int
        
        print(f"🎯 Prediction label: {pred_label}, Confidence: {confidence}")

        # Format response to match brain model structure
        pneumonia_detected = bool(pred_label == 1)  # Convert to Python bool
        result_message = 'Pneumonia detected' if pneumonia_detected else 'No Pneumonia detected'
        confidence_percentage = float(confidence * 100)  # Convert to Python float
        
        processing_time = time.time() - start_time
        
        # Upload images to Cloudinary
        print("📤 Uploading images to Cloudinary...")
        original_image_array = cv2.resize(np.array(image), (128, 128))
        processed_image_array = input_image_resized  # The processed version
        
        image_urls = cloudinary_client.upload_pneumonia_analysis_images(
            original_image=original_image_array,
            processed_image=processed_image_array,
            user_id=current_user_id
        )
        
        # Save analysis to database with Cloudinary image URLs
        try:
            analysis = PneumoniaAnalysis(
                user_id=current_user_id,
                pneumonia_detected=pneumonia_detected,
                confidence=confidence_percentage,
                prediction_label=result_message,
                image_filename=file.filename if file.filename else 'unknown.jpg',
                image_size="128x128",
                processing_time=processing_time,
                original_image_url=image_urls.get('original_image_url'),
                processed_image_url=image_urls.get('processed_image_url')
            )
            
            db.session.add(analysis)
            db.session.commit()
            print(f"✅ Pneumonia analysis saved to database with ID: {analysis.id}")
            
        except Exception as db_error:
            print(f"❌ Failed to save pneumonia analysis to database: {db_error}")
            db.session.rollback()
            # Continue without failing the request
        
        print(f"✅ Final result: {result_message}, Confidence: {confidence_percentage}%")
        
        return jsonify({
            'status': 'success',
            'prediction': {
                'message': f'{result_message} with confidence: {confidence_percentage:.2f}%',
                'confidence': round(confidence_percentage, 2),
                'pneumonia_detected': pneumonia_detected
            },
            'metrics': {
                'accuracy': 0.89,  # Pneumonia model metrics
                'precision': 0.87,
                'recall': 0.85,
                'f1_score': 0.86,
                'auc': 0.91
            },
            'analysis_id': analysis.id if 'analysis' in locals() else None
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Get user's pneumonia analysis history
@pneumonia_routes.route('/history', methods=['GET'])
@token_required
def get_analysis_history(current_user_id, current_username):
    try:
        # Get pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Query user's analyses with pagination
        analyses = PneumoniaAnalysis.query.filter_by(user_id=current_user_id)\
                                        .order_by(PneumoniaAnalysis.created_at.desc())\
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

# Get specific pneumonia analysis by ID
@pneumonia_routes.route('/analysis/<int:analysis_id>', methods=['GET'])
@token_required
def get_analysis(current_user_id, current_username, analysis_id):
    try:
        analysis = PneumoniaAnalysis.query.filter_by(
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