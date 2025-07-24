"""
Migration script to add multiple image URL columns using Flask app context
"""

import os
import sys
from flask import Flask
from models.database import db
from dotenv import load_dotenv

load_dotenv()

def create_app():
    """Create Flask app for migration"""
    app = Flask(__name__)
    
    # Database configuration
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable must be set")
    
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    return app

def run_migration():
    """Add image URL columns to existing tables"""
    
    app = create_app()
    
    with app.app_context():
        try:
            print("🔄 Starting database migration...")
            
            # Get database connection
            connection = db.engine.connect()
            trans = connection.begin()
            
            try:
                # Add image URL columns to brain_analyses table
                print("📊 Adding image URL columns to brain_analyses table...")
                
                try:
                    connection.execute(db.text("ALTER TABLE brain_analyses ADD COLUMN original_image_url VARCHAR(500)"))
                    print("✅ Added original_image_url to brain_analyses")
                except Exception as e:
                    if "already exists" in str(e) or "duplicate column" in str(e).lower():
                        print("ℹ️ original_image_url already exists in brain_analyses")
                    else:
                        raise e
                
                try:
                    connection.execute(db.text("ALTER TABLE brain_analyses ADD COLUMN cropped_image_url VARCHAR(500)"))
                    print("✅ Added cropped_image_url to brain_analyses")
                except Exception as e:
                    if "already exists" in str(e) or "duplicate column" in str(e).lower():
                        print("ℹ️ cropped_image_url already exists in brain_analyses")
                    else:
                        raise e
                
                try:
                    connection.execute(db.text("ALTER TABLE brain_analyses ADD COLUMN threshold_image_url VARCHAR(500)"))
                    print("✅ Added threshold_image_url to brain_analyses")
                except Exception as e:
                    if "already exists" in str(e) or "duplicate column" in str(e).lower():
                        print("ℹ️ threshold_image_url already exists in brain_analyses")
                    else:
                        raise e
                
                # Add image URL columns to pneumonia_analyses table
                print("📊 Adding image URL columns to pneumonia_analyses table...")
                
                try:
                    connection.execute(db.text("ALTER TABLE pneumonia_analyses ADD COLUMN original_image_url VARCHAR(500)"))
                    print("✅ Added original_image_url to pneumonia_analyses")
                except Exception as e:
                    if "already exists" in str(e) or "duplicate column" in str(e).lower():
                        print("ℹ️ original_image_url already exists in pneumonia_analyses")
                    else:
                        raise e
                
                try:
                    connection.execute(db.text("ALTER TABLE pneumonia_analyses ADD COLUMN processed_image_url VARCHAR(500)"))
                    print("✅ Added processed_image_url to pneumonia_analyses")
                except Exception as e:
                    if "already exists" in str(e) or "duplicate column" in str(e).lower():
                        print("ℹ️ processed_image_url already exists in pneumonia_analyses")
                    else:
                        raise e
                
                # Commit the transaction
                trans.commit()
                print("✅ Database migration completed successfully!")
                
            except Exception as e:
                trans.rollback()
                print(f"❌ Migration failed: {e}")
                return False
            finally:
                connection.close()
                
            return True
            
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False

if __name__ == "__main__":
    print("🚀 Starting database migration for Cloudinary image URLs...")
    
    if run_migration():
        print("\n🎉 Migration completed successfully!")
        print("You can now run your server with Cloudinary image storage.")
        print("\n📋 Next steps:")
        print("1. Run: python server.py")
        print("2. Test the endpoints - images will be stored in Cloudinary")
        print("3. Check history - all analyses will include multiple image URLs")
        print("\n🖼️ Brain Analysis will store: original, cropped, threshold images")
        print("🫁 Pneumonia Analysis will store: original, processed images")
    else:
        print("\n❌ Migration failed. Please check the error messages above.")