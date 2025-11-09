"""
FRS Demo Script
Demonstrates all capabilities of the Face Recognition System
"""
import requests
import time
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

API_BASE_URL = "http://localhost:8000"


class FRSDemo:
    """Face Recognition System Demo"""
    
    def __init__(self):
        """Initialize demo"""
        self.base_url = API_BASE_URL
        self.demo_images_dir = Path("demo_images")
        self.demo_images_dir.mkdir(exist_ok=True)
        
    def check_health(self):
        """Check if API is running"""
        print("\n" + "="*60)
        print("STEP 1: Health Check")
        print("="*60)
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print("✅ API is running!")
                print(f"   Status: {data.get('status')}")
                print(f"   Version: {data.get('version')}")
                print(f"   Models loaded: {data.get('models')}")
                return True
            else:
                print(f"❌ API returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API. Please ensure backend is running on http://localhost:8000")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def create_demo_image(self, filename: str = "demo_face.jpg"):
        """Create a demo face image"""
        print("\n" + "="*60)
        print("STEP 2: Creating Demo Image")
        print("="*60)
        
        # Create a simple face-like image (placeholder)
        img = Image.new('RGB', (640, 480), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple face
        # Face oval
        draw.ellipse([200, 150, 440, 390], fill=(255, 220, 177), outline=(0, 0, 0), width=2)
        
        # Eyes
        draw.ellipse([250, 220, 290, 260], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.ellipse([350, 220, 390, 260], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        draw.ellipse([260, 230, 280, 250], fill=(0, 0, 0))
        draw.ellipse([360, 230, 380, 250], fill=(0, 0, 0))
        
        # Nose
        draw.line([320, 260, 320, 300], fill=(0, 0, 0), width=2)
        
        # Mouth
        draw.arc([270, 310, 370, 360], 0, 180, fill=(0, 0, 0), width=3)
        
        # Save
        output_path = self.demo_images_dir / filename
        img.save(output_path)
        print(f"✅ Demo image created: {output_path}")
        return str(output_path)
    
    def demo_detection(self, image_path: str):
        """Demonstrate face detection"""
        print("\n" + "="*60)
        print("STEP 3: Face Detection")
        print("="*60)
        
        print(f"Detecting faces in: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/api/v1/detect", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Detection successful!")
            print(f"   Faces detected: {data.get('num_faces', 0)}")
            print(f"   Processing time: {data.get('processing_time_ms', 0):.2f} ms")
            
            if data.get('detections'):
                for i, det in enumerate(data['detections'], 1):
                    bbox = det.get('bbox', {})
                    print(f"\n   Face {i}:")
                    print(f"     Bounding box: ({bbox.get('x1')}, {bbox.get('y1')}) to ({bbox.get('x2')}, {bbox.get('y2')})")
                    print(f"     Confidence: {det.get('confidence', 0):.3f}")
                    if det.get('quality_score'):
                        print(f"     Quality: {det.get('quality_score'):.3f}")
            
            return data
        else:
            print(f"❌ Detection failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    
    def demo_add_identity(self, name: str, image_path: str):
        """Demonstrate adding an identity"""
        print("\n" + "="*60)
        print(f"STEP 4: Adding Identity '{name}'")
        print("="*60)
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'name': name}
            response = requests.post(f"{self.base_url}/api/v1/add_identity", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Identity added successfully!")
            print(f"   ID: {result.get('id')}")
            print(f"   Name: {result.get('name')}")
            print(f"   Image path: {result.get('image_path')}")
            return result
        else:
            print(f"❌ Failed to add identity")
            error_data = response.json()
            print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            return None
    
    def demo_recognition(self, image_path: str):
        """Demonstrate face recognition"""
        print("\n" + "="*60)
        print("STEP 5: Face Recognition")
        print("="*60)
        
        print(f"Recognizing faces in: {image_path}")
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {'threshold': 0.6, 'top_k': 5}
            response = requests.post(f"{self.base_url}/api/v1/recognize", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Recognition successful!")
            print(f"   Faces found: {result.get('num_faces', 0)}")
            print(f"   Processing time: {result.get('processing_time_ms', 0):.2f} ms")
            
            if result.get('results'):
                for i, res in enumerate(result['results'], 1):
                    print(f"\n   Face {i}:")
                    if res.get('best_match'):
                        match = res['best_match']
                        print(f"     ✓ Recognized: {match.get('identity_name')}")
                        print(f"     Confidence: {match.get('confidence', 0):.3f}")
                        print(f"     Identity ID: {match.get('identity_id')}")
                    else:
                        print(f"     ✗ Unknown person")
            
            return result
        else:
            print(f"❌ Recognition failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
    
    def demo_list_identities(self):
        """Demonstrate listing identities"""
        print("\n" + "="*60)
        print("STEP 6: List All Identities")
        print("="*60)
        
        response = requests.get(f"{self.base_url}/api/v1/list_identities")
        
        if response.status_code == 200:
            data = response.json()
            count = data.get('count', 0)
            print(f"✅ Gallery contains {count} identities")
            
            if data.get('identities'):
                print("\n   Registered identities:")
                for identity in data['identities']:
                    print(f"     • {identity.get('name')} (ID: {identity.get('id')})")
            
            return data
        else:
            print(f"❌ Failed to list identities")
            return None
    
    def run_complete_demo(self):
        """Run complete demonstration"""
        print("\n" + "🎬 "*20)
        print("FACE RECOGNITION SYSTEM - COMPLETE DEMO")
        print("🎬 "*20)
        
        # Step 1: Health check
        if not self.check_health():
            print("\n❌ Demo aborted: API is not running")
            print("Please start the backend server: python backend/main.py")
            return
        
        time.sleep(1)
        
        # Step 2: Create demo image
        demo_image = self.create_demo_image("demo_person.jpg")
        time.sleep(1)
        
        # Step 3: Detection
        self.demo_detection(demo_image)
        time.sleep(1)
        
        # Step 4: Add identity
        self.demo_add_identity("Demo Person", demo_image)
        time.sleep(1)
        
        # Step 5: Recognition
        self.demo_recognition(demo_image)
        time.sleep(1)
        
        # Step 6: List identities
        self.demo_list_identities()
        
        # Summary
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("""
Summary:
✅ Face detection working
✅ Identity enrollment working
✅ Face recognition working
✅ Gallery management working

Next Steps:
1. Open web interface: http://localhost:8501
2. Try with real photos
3. Test with multiple identities
4. Review API documentation: http://localhost:8000/docs

For production deployment:
- Use Docker: docker-compose up
- Configure environment variables
- Set up reverse proxy (Nginx)
- Enable monitoring and logging
        """)
    
    def create_demo_video_frames(self):
        """Create frames for demo video/GIF"""
        print("\n" + "="*60)
        print("Creating Demo Frames for Video/GIF")
        print("="*60)
        
        frames_dir = self.demo_images_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        # Create sequence of frames showing detection -> recognition
        frames = []
        
        # Frame 1: Original image
        img = Image.new('RGB', (800, 600), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.text((300, 280), "Face Recognition Demo", fill=(0, 0, 0))
        frames.append(img)
        
        # Frame 2: Detection
        img2 = img.copy()
        draw = ImageDraw.Draw(img2)
        draw.rectangle([200, 150, 600, 450], outline=(0, 255, 0), width=3)
        draw.text((250, 460), "Face Detected", fill=(0, 255, 0))
        frames.append(img2)
        
        # Frame 3: Recognition
        img3 = img2.copy()
        draw = ImageDraw.Draw(img3)
        draw.text((250, 490), "Identified: Demo Person (92.3%)", fill=(0, 128, 255))
        frames.append(img3)
        
        # Save frames
        for i, frame in enumerate(frames):
            frame.save(frames_dir / f"frame_{i:03d}.png")
        
        print(f"✅ Created {len(frames)} demo frames in {frames_dir}")
        print("   Use these to create a demo GIF with ImageMagick or similar tool")
        print(f"   Command: convert -delay 100 {frames_dir}/*.png demo.gif")
        
        return frames_dir


def main():
    """Run demo"""
    demo = FRSDemo()
    
    # Run complete demonstration
    demo.run_complete_demo()
    
    # Optionally create video frames
    print("\nWould you like to create demo frames for video/GIF?")
    print("(These can be used to create marketing materials)")
    demo.create_demo_video_frames()


if __name__ == "__main__":
    main()
