"""
Demo script for Face Recognition Service
Demonstrates core functionality with sample data
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import requests
from pathlib import Path
import time
import json


BASE_URL = "http://localhost:8000"


def check_service():
    """Check if service is running"""
    print("Checking service status...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Service is {data['status']}")
            print(f"  App: {data['app_name']} v{data['version']}")
            return True
        else:
            print("✗ Service returned error")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Service not reachable: {e}")
        print("\nPlease start the service first:")
        print("  python main.py")
        return False


def demo_detection(image_path: str):
    """Demo face detection"""
    print(f"\n{'='*60}")
    print("DEMO: Face Detection")
    print('='*60)
    
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return
    
    print(f"Detecting faces in: {image_path}")
    
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/api/v1/detect",
            files={'file': f}
        )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Detection completed in {data['processing_time_ms']:.2f}ms")
        print(f"  Found {data['num_faces']} face(s)")
        
        for i, det in enumerate(data['detections'], 1):
            bbox = det['bbox']
            print(f"\n  Face {i}:")
            print(f"    BBox: ({bbox['x1']:.1f}, {bbox['y1']:.1f}) - ({bbox['x2']:.1f}, {bbox['y2']:.1f})")
            print(f"    Confidence: {det['confidence']:.3f}")
            if det.get('quality_score'):
                print(f"    Quality: {det['quality_score']:.3f}")
    else:
        print(f"✗ Detection failed: {response.text}")


def demo_add_identity(name: str, image_path: str, metadata: dict = None):
    """Demo adding identity to gallery"""
    print(f"\n{'='*60}")
    print("DEMO: Add Identity")
    print('='*60)
    
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return None
    
    print(f"Adding identity: {name}")
    print(f"Image: {image_path}")
    
    data = {'name': name}
    if metadata:
        data['metadata'] = json.dumps(metadata)
    
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/api/v1/add_identity",
            files={'file': f},
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Identity added successfully")
        print(f"  ID: {result['id']}")
        print(f"  Name: {result['name']}")
        print(f"  Image: {result['image_path']}")
        return result['id']
    else:
        error = response.json()
        print(f"✗ Failed to add identity: {error.get('detail', 'Unknown error')}")
        return None


def demo_recognition(image_path: str):
    """Demo face recognition"""
    print(f"\n{'='*60}")
    print("DEMO: Face Recognition")
    print('='*60)
    
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return
    
    print(f"Recognizing faces in: {image_path}")
    
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{BASE_URL}/api/v1/recognize",
            files={'file': f}
        )
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Recognition completed in {data['processing_time_ms']:.2f}ms")
        print(f"  Found {data['num_faces']} face(s)")
        
        for i, result in enumerate(data['results'], 1):
            print(f"\n  Face {i}:")
            print(f"    Detection confidence: {result['detection_confidence']:.3f}")
            
            if result['best_match']:
                match = result['best_match']
                print(f"    ✓ Recognized: {match['identity_name']}")
                print(f"      Similarity: {match['similarity']:.3f}")
                print(f"      Identity ID: {match['identity_id']}")
            else:
                print(f"    ✗ No match found")
            
            if len(result['matches']) > 1:
                print(f"    Top {len(result['matches'])} matches:")
                for match in result['matches']:
                    print(f"      - {match['identity_name']}: {match['similarity']:.3f}")
    else:
        print(f"✗ Recognition failed: {response.text}")


def demo_list_identities():
    """Demo listing identities"""
    print(f"\n{'='*60}")
    print("DEMO: List Identities")
    print('='*60)
    
    response = requests.get(f"{BASE_URL}/api/v1/list_identities")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Gallery contains {data['count']} identity(ies)")
        
        for identity in data['identities']:
            print(f"\n  ID {identity['id']}: {identity['name']}")
            print(f"    Image: {identity['image_path']}")
            if identity.get('metadata'):
                print(f"    Metadata: {identity['metadata']}")
            print(f"    Added: {identity['created_at']}")
    else:
        print(f"✗ Failed to list identities: {response.text}")


def demo_stats():
    """Demo system statistics"""
    print(f"\n{'='*60}")
    print("DEMO: System Statistics")
    print('='*60)
    
    response = requests.get(f"{BASE_URL}/stats")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ System Statistics:")
        print(f"  Identities in gallery: {data['num_identities']}")
        print(f"  Total embeddings: {data['num_embeddings']}")
        print(f"  Total detections: {data['total_detections']}")
        print(f"  Uptime: {data['uptime_seconds']:.1f} seconds")
        
        if 'matcher_stats' in data:
            print(f"\n  Matcher Configuration:")
            stats = data['matcher_stats']
            print(f"    Using Faiss: {stats.get('use_faiss', False)}")
            print(f"    Embedding dimension: {stats.get('embedding_dim', 0)}")
    else:
        print(f"✗ Failed to get stats: {response.text}")


def run_full_demo():
    """Run complete demo workflow"""
    print("\n" + "="*60)
    print(" "*15 + "FACE RECOGNITION SERVICE DEMO")
    print("="*60)
    
    # Check service
    if not check_service():
        return
    
    # Show initial stats
    demo_stats()
    
    # List existing identities
    demo_list_identities()
    
    print("\n\nFor full demo, you need sample images.")
    print("Place images in a 'demo_images' directory:")
    print("  - demo_images/person1.jpg (for adding to gallery)")
    print("  - demo_images/test.jpg (for recognition)")
    print("\nThen run specific demos:")
    print("  python demo.py --detect demo_images/test.jpg")
    print("  python demo.py --add-identity 'John Doe' demo_images/person1.jpg")
    print("  python demo.py --recognize demo_images/test.jpg")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Service Demo")
    parser.add_argument("--detect", type=str, help="Run face detection demo")
    parser.add_argument("--add-identity", nargs=2, metavar=('NAME', 'IMAGE'), 
                       help="Add identity to gallery")
    parser.add_argument("--recognize", type=str, help="Run face recognition demo")
    parser.add_argument("--list", action="store_true", help="List all identities")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--full", action="store_true", help="Run full demo")
    
    args = parser.parse_args()
    
    # Check service first
    if not check_service():
        return
    
    # Run requested demo
    if args.detect:
        demo_detection(args.detect)
    elif args.add_identity:
        name, image = args.add_identity
        demo_add_identity(name, image, metadata={"demo": True})
    elif args.recognize:
        demo_recognition(args.recognize)
    elif args.list:
        demo_list_identities()
    elif args.stats:
        demo_stats()
    elif args.full:
        run_full_demo()
    else:
        run_full_demo()


if __name__ == "__main__":
    main()
