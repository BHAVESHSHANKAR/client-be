from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    brain_analyses = db.relationship('BrainAnalysis', backref='user', lazy=True, cascade='all, delete-orphan')
    pneumonia_analyses = db.relationship('PneumoniaAnalysis', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set the password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if provided password matches the hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'age': self.age,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<User {self.username}>'

class BrainAnalysis(db.Model):
    __tablename__ = 'brain_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Analysis results
    tumor_detected = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    
    # Model metrics at time of analysis
    model_accuracy = db.Column(db.Float)
    model_loss = db.Column(db.Float)
    model_precision = db.Column(db.Float)
    model_recall = db.Column(db.Float)
    model_auc = db.Column(db.Float)
    
    # Image processing info
    image_filename = db.Column(db.String(255))
    image_size = db.Column(db.String(50))  # e.g., "128x128"
    processing_time = db.Column(db.Float)  # in seconds
    
    # Cloudinary image URLs
    original_image_url = db.Column(db.String(500))
    cropped_image_url = db.Column(db.String(500))
    threshold_image_url = db.Column(db.String(500))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert analysis object to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'tumor_detected': self.tumor_detected,
            'confidence': self.confidence,
            'model_metrics': {
                'accuracy': self.model_accuracy,
                'loss': self.model_loss,
                'precision': self.model_precision,
                'recall': self.model_recall,
                'auc': self.model_auc
            },
            'image_info': {
                'filename': self.image_filename,
                'size': self.image_size,
                'processing_time': self.processing_time
            },
            'images': {
                'original_image_url': self.original_image_url,
                'cropped_image_url': self.cropped_image_url,
                'threshold_image_url': self.threshold_image_url
            },
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<BrainAnalysis {self.id} - User {self.user_id}>'

class PneumoniaAnalysis(db.Model):
    __tablename__ = 'pneumonia_analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Analysis results
    pneumonia_detected = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    prediction_label = db.Column(db.String(50), nullable=False)  # "Pneumonia detected" or "No Pneumonia detected"
    
    # Image processing info
    image_filename = db.Column(db.String(255))
    image_size = db.Column(db.String(50))  # e.g., "128x128"
    processing_time = db.Column(db.Float)  # in seconds
    
    # Cloudinary image URLs
    original_image_url = db.Column(db.String(500))
    processed_image_url = db.Column(db.String(500))
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert analysis object to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'pneumonia_detected': self.pneumonia_detected,
            'confidence': self.confidence,
            'prediction_label': self.prediction_label,
            'image_info': {
                'filename': self.image_filename,
                'size': self.image_size,
                'processing_time': self.processing_time
            },
            'images': {
                'original_image_url': self.original_image_url,
                'processed_image_url': self.processed_image_url
            },
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    def __repr__(self):
        return f'<PneumoniaAnalysis {self.id} - User {self.user_id}>'

def init_db(app):
    """Initialize database with Flask app"""
    db.init_app(app)
    
    with app.app_context():
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully")