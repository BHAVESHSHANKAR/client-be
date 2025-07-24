"""
Database migration script to add Cloudinary image URL columns
Run this script to update existing database tables with new image URL columns
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def run_migration():
    """Run database migration to add image URL columns"""
    
    if not DATABASE_URL:
        print("❌ DATABASE_URL not found in environment variables")
        return False
    
    try:
        # Create database engine
        engine = create_engine(DATABASE_URL)
        
        print("🔄 Starting database migration...")
        
        with engine.connect() as connection:
            # Start a transaction
            trans = connection.begin()
            
            try:
                # Add columns to brain_analyses table
                print("📊 Adding image URL columns to brain_analyses table...")
                
                # Check if columns already exist
                result = connection.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'brain_analyses' 
                    AND column_name IN ('original_image_url', 'cropped_image_url', 'threshold_image_url')
                """))
                
                existing_brain_columns = [row[0] for row in result]
                
                if 'original_image_url' not in existing_brain_columns:
                    connection.execute(text("ALTER TABLE brain_analyses ADD COLUMN original_image_url VARCHAR(500)"))
                    print("✅ Added original_image_url to brain_analyses")
                else:
                    print("ℹ️ original_image_url already exists in brain_analyses")
                
                if 'cropped_image_url' not in existing_brain_columns:
                    connection.execute(text("ALTER TABLE brain_analyses ADD COLUMN cropped_image_url VARCHAR(500)"))
                    print("✅ Added cropped_image_url to brain_analyses")
                else:
                    print("ℹ️ cropped_image_url already exists in brain_analyses")
                
                if 'threshold_image_url' not in existing_brain_columns:
                    connection.execute(text("ALTER TABLE brain_analyses ADD COLUMN threshold_image_url VARCHAR(500)"))
                    print("✅ Added threshold_image_url to brain_analyses")
                else:
                    print("ℹ️ threshold_image_url already exists in brain_analyses")
                
                # Add columns to pneumonia_analyses table
                print("📊 Adding image URL columns to pneumonia_analyses table...")
                
                # Check if columns already exist
                result = connection.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'pneumonia_analyses' 
                    AND column_name IN ('original_image_url', 'processed_image_url')
                """))
                
                existing_pneumonia_columns = [row[0] for row in result]
                
                if 'original_image_url' not in existing_pneumonia_columns:
                    connection.execute(text("ALTER TABLE pneumonia_analyses ADD COLUMN original_image_url VARCHAR(500)"))
                    print("✅ Added original_image_url to pneumonia_analyses")
                else:
                    print("ℹ️ original_image_url already exists in pneumonia_analyses")
                
                if 'processed_image_url' not in existing_pneumonia_columns:
                    connection.execute(text("ALTER TABLE pneumonia_analyses ADD COLUMN processed_image_url VARCHAR(500)"))
                    print("✅ Added processed_image_url to pneumonia_analyses")
                else:
                    print("ℹ️ processed_image_url already exists in pneumonia_analyses")
                
                # Commit the transaction
                trans.commit()
                print("✅ Database migration completed successfully!")
                return True
                
            except Exception as e:
                # Rollback on error
                trans.rollback()
                print(f"❌ Migration failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def verify_migration():
    """Verify that the migration was successful"""
    
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as connection:
            # Check brain_analyses table
            result = connection.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'brain_analyses' 
                AND column_name IN ('original_image_url', 'cropped_image_url', 'threshold_image_url')
                ORDER BY column_name
            """))
            
            brain_columns = [row[0] for row in result]
            print(f"🧠 Brain analysis image columns: {brain_columns}")
            
            # Check pneumonia_analyses table
            result = connection.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'pneumonia_analyses' 
                AND column_name IN ('original_image_url', 'processed_image_url')
                ORDER BY column_name
            """))
            
            pneumonia_columns = [row[0] for row in result]
            print(f"🫁 Pneumonia analysis image columns: {pneumonia_columns}")
            
            # Verify all columns exist
            expected_brain = ['cropped_image_url', 'original_image_url', 'threshold_image_url']
            expected_pneumonia = ['original_image_url', 'processed_image_url']
            
            brain_success = all(col in brain_columns for col in expected_brain)
            pneumonia_success = all(col in pneumonia_columns for col in expected_pneumonia)
            
            if brain_success and pneumonia_success:
                print("✅ Migration verification successful - all columns present!")
                return True
            else:
                print("❌ Migration verification failed - some columns missing")
                return False
                
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting database migration for Cloudinary image URLs...")
    
    # Run migration
    if run_migration():
        print("\n🔍 Verifying migration...")
        if verify_migration():
            print("\n🎉 Migration completed successfully!")
            print("You can now run your server with Cloudinary image storage.")
        else:
            print("\n⚠️ Migration completed but verification failed.")
    else:
        print("\n❌ Migration failed. Please check the error messages above.")