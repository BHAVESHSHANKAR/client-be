import cloudinary
import cloudinary.uploader
import cloudinary.api
import cv2
import numpy as np
from PIL import Image
import io
import base64
from os import getenv
from dotenv import load_dotenv

load_dotenv()

# Configure Cloudinary
cloudinary.config(
    cloud_name=getenv("CLOUD_NAME"),
    api_key=getenv("CLOUD_API_KEY"),
    api_secret=getenv("CLOUD_API_SECRET"),
    secure=True
)

class CloudinaryClient:
    def __init__(self):
        self.folder_name = "uploaded_ml_images"
        
    def upload_image(self, image_data, filename="image.png", image_type="analysis"):
        """
        Upload image to Cloudinary
        
        Args:
            image_data: Image data (numpy array, PIL Image, or bytes)
            filename: Name for the uploaded file (without extension)
            image_type: Type of image (original, cropped, threshold, processed)
            
        Returns:
            dict: Upload response with URL and file info
        """
        try:
            # Convert image data to bytes if needed
            if isinstance(image_data, np.ndarray):
                # Convert numpy array to bytes
                _, buffer = cv2.imencode('.png', image_data)
                image_bytes = buffer.tobytes()
            elif isinstance(image_data, Image.Image):
                # Convert PIL Image to bytes
                buffer = io.BytesIO()
                image_data.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
            elif isinstance(image_data, bytes):
                image_bytes = image_data
            else:
                raise ValueError("Unsupported image data type")
            
            # Create a unique public_id (fix double folder issue)
            public_id = f"{filename}_{image_type}"
            
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(
                image_bytes,
                public_id=public_id,
                folder=self.folder_name,
                resource_type="image",
                format="png",
                overwrite=True,
                transformation=[
                    {"quality": "auto"},
                    {"fetch_format": "auto"}
                ]
            )
            
            result = {
                "url": upload_result.get("secure_url", ""),
                "public_id": upload_result.get("public_id", ""),
                "name": filename,
                "size": len(image_bytes),
                "format": upload_result.get("format", "png"),
                "width": upload_result.get("width", 0),
                "height": upload_result.get("height", 0)
            }
            
            print(f"✅ Image uploaded to Cloudinary: {result['url']}")
            return result
            
        except Exception as e:
            print(f"❌ Cloudinary upload error: {e}")
            return None
    
    def upload_brain_analysis_images(self, original_image, cropped_image, threshold_image, user_id):
        """
        Upload all brain analysis images
        
        Args:
            original_image: Original MRI image
            cropped_image: Cropped brain region
            threshold_image: Threshold processed image
            user_id: User ID for unique naming
            
        Returns:
            dict: URLs for all uploaded images
        """
        import time
        timestamp = int(time.time())
        base_filename = f"brain_{user_id}_{timestamp}"
        
        results = {}
        
        # Upload original image
        original_result = self.upload_image(
            original_image, 
            base_filename, 
            "original"
        )
        results['original_image_url'] = original_result['url'] if original_result else None
        
        # Upload cropped image
        cropped_result = self.upload_image(
            cropped_image, 
            base_filename, 
            "cropped"
        )
        results['cropped_image_url'] = cropped_result['url'] if cropped_result else None
        
        # Upload threshold image
        threshold_result = self.upload_image(
            threshold_image, 
            base_filename, 
            "threshold"
        )
        results['threshold_image_url'] = threshold_result['url'] if threshold_result else None
        
        return results
    
    def upload_pneumonia_analysis_images(self, original_image, processed_image, user_id):
        """
        Upload all pneumonia analysis images
        
        Args:
            original_image: Original chest X-ray image
            processed_image: Processed image for analysis
            user_id: User ID for unique naming
            
        Returns:
            dict: URLs for all uploaded images
        """
        import time
        timestamp = int(time.time())
        base_filename = f"pneumonia_{user_id}_{timestamp}"
        
        results = {}
        
        # Upload original image
        original_result = self.upload_image(
            original_image, 
            base_filename, 
            "original"
        )
        results['original_image_url'] = original_result['url'] if original_result else None
        
        # Upload processed image
        processed_result = self.upload_image(
            processed_image, 
            base_filename, 
            "processed"
        )
        results['processed_image_url'] = processed_result['url'] if processed_result else None
        
        return results
    
    def delete_image(self, public_id):
        """
        Delete image from Cloudinary
        
        Args:
            public_id: Public ID of the image to delete
            
        Returns:
            bool: Success status
        """
        try:
            result = cloudinary.uploader.destroy(public_id)
            return result.get('result') == 'ok'
        except Exception as e:
            print(f"❌ Error deleting image: {e}")
            return False
    
    def get_image_info(self, public_id):
        """
        Get information about an uploaded image
        
        Args:
            public_id: Public ID of the image
            
        Returns:
            dict: Image information
        """
        try:
            result = cloudinary.api.resource(public_id)
            return {
                "url": result.get("secure_url", ""),
                "format": result.get("format", ""),
                "width": result.get("width", 0),
                "height": result.get("height", 0),
                "size": result.get("bytes", 0),
                "created_at": result.get("created_at", "")
            }
        except Exception as e:
            print(f"❌ Error getting image info: {e}")
            return None

# Global instance
cloudinary_client = CloudinaryClient()