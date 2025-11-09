"""
Evaluation script for Face Recognition System
Calculates precision, recall, identification rate, and other metrics
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

from models.face_detector import FaceDetector
from models.feature_extractor import FeatureExtractor
from models.matcher import FaceMatcher
from utils.image_processing import load_image, crop_face
from utils.logger import log


def evaluate_detection(detector: FaceDetector, test_dir: str):
    """
    Evaluate face detection metrics
    
    Args:
        detector: Face detector instance
        test_dir: Directory with test images
    """
    print("\n" + "="*50)
    print("Evaluating Face Detection")
    print("="*50)
    
    test_path = Path(test_dir)
    image_files = list(test_path.rglob("*.jpg")) + list(test_path.rglob("*.png"))
    
    total_images = len(image_files)
    images_with_faces = 0
    total_faces = 0
    
    for image_file in tqdm(image_files, desc="Detection evaluation"):
        try:
            image = load_image(str(image_file))
            boxes, _, _ = detector.detect(image)
            
            if len(boxes) > 0:
                images_with_faces += 1
                total_faces += len(boxes)
        except Exception as e:
            log.error(f"Error processing {image_file}: {str(e)}")
    
    detection_rate = images_with_faces / total_images if total_images > 0 else 0
    avg_faces = total_faces / images_with_faces if images_with_faces > 0 else 0
    
    results = {
        "total_images": total_images,
        "images_with_faces": images_with_faces,
        "total_faces_detected": total_faces,
        "detection_rate": detection_rate,
        "avg_faces_per_image": avg_faces
    }
    
    print(f"\nResults:")
    print(f"  Total images: {total_images}")
    print(f"  Images with faces: {images_with_faces}")
    print(f"  Detection rate: {detection_rate*100:.2f}%")
    print(f"  Total faces detected: {total_faces}")
    print(f"  Avg faces per image: {avg_faces:.2f}")
    
    return results


def evaluate_recognition(
    detector: FaceDetector,
    extractor: FeatureExtractor,
    matcher: FaceMatcher,
    gallery_dir: str,
    test_dir: str,
    threshold: float = 0.6
):
    """
    Evaluate face recognition metrics
    
    Args:
        detector: Face detector
        extractor: Feature extractor
        matcher: Face matcher
        gallery_dir: Gallery directory (one folder per identity)
        test_dir: Test directory (one folder per identity)
        threshold: Recognition threshold
    """
    print("\n" + "="*50)
    print("Evaluating Face Recognition")
    print("="*50)
    
    # Build gallery
    print("\nBuilding gallery...")
    gallery_path = Path(gallery_dir)
    identity_to_id = {}
    current_id = 0
    
    for identity_dir in tqdm(list(gallery_path.iterdir()), desc="Gallery"):
        if not identity_dir.is_dir():
            continue
        
        identity_name = identity_dir.name
        identity_to_id[identity_name] = current_id
        
        # Process all images for this identity
        image_files = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        
        for image_file in image_files:
            try:
                image = load_image(str(image_file))
                boxes, _, _ = detector.detect(image)
                
                if len(boxes) == 0:
                    continue
                
                face = crop_face(image, boxes[0])
                embedding = extractor.extract(face)
                matcher.add_embedding(embedding, current_id, identity_name)
                
            except Exception as e:
                log.error(f"Error processing gallery image {image_file}: {str(e)}")
        
        current_id += 1
    
    print(f"Gallery built with {len(identity_to_id)} identities")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_path = Path(test_dir)
    
    results = {
        "total": 0,
        "correct_top1": 0,
        "correct_top5": 0,
        "no_match": 0,
        "per_identity": defaultdict(lambda: {"total": 0, "correct": 0})
    }
    
    for identity_dir in tqdm(list(test_path.iterdir()), desc="Test"):
        if not identity_dir.is_dir():
            continue
        
        identity_name = identity_dir.name
        
        if identity_name not in identity_to_id:
            log.warning(f"Identity {identity_name} not in gallery, skipping")
            continue
        
        true_id = identity_to_id[identity_name]
        image_files = list(identity_dir.glob("*.jpg")) + list(identity_dir.glob("*.png"))
        
        for image_file in image_files:
            try:
                image = load_image(str(image_file))
                boxes, _, _ = detector.detect(image)
                
                if len(boxes) == 0:
                    continue
                
                face = crop_face(image, boxes[0])
                embedding = extractor.extract(face)
                matches = matcher.match(embedding, threshold=threshold, top_k=5)
                
                results["total"] += 1
                results["per_identity"][identity_name]["total"] += 1
                
                if len(matches) == 0:
                    results["no_match"] += 1
                else:
                    # Check top-1
                    if matches[0][0] == true_id:
                        results["correct_top1"] += 1
                        results["per_identity"][identity_name]["correct"] += 1
                    
                    # Check top-5
                    if any(m[0] == true_id for m in matches[:5]):
                        results["correct_top5"] += 1
                
            except Exception as e:
                log.error(f"Error processing test image {image_file}: {str(e)}")
    
    # Calculate metrics
    total = results["total"]
    if total > 0:
        accuracy_top1 = results["correct_top1"] / total
        accuracy_top5 = results["correct_top5"] / total
        no_match_rate = results["no_match"] / total
    else:
        accuracy_top1 = accuracy_top5 = no_match_rate = 0
    
    metrics = {
        "total_tests": total,
        "accuracy_top1": accuracy_top1,
        "accuracy_top5": accuracy_top5,
        "no_match_rate": no_match_rate,
        "correct_top1": results["correct_top1"],
        "correct_top5": results["correct_top5"],
        "threshold": threshold
    }
    
    print(f"\nResults:")
    print(f"  Total tests: {total}")
    print(f"  Top-1 Accuracy: {accuracy_top1*100:.2f}%")
    print(f"  Top-5 Accuracy: {accuracy_top5*100:.2f}%")
    print(f"  No match rate: {no_match_rate*100:.2f}%")
    
    # Per-identity accuracy
    print(f"\nPer-identity accuracy:")
    for identity_name, stats in sorted(results["per_identity"].items()):
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"  {identity_name}: {acc*100:.2f}% ({stats['correct']}/{stats['total']})")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate Face Recognition System")
    parser.add_argument("--gallery", type=str, required=True, help="Gallery directory")
    parser.add_argument("--test", type=str, required=True, help="Test directory")
    parser.add_argument("--eval-detection", action="store_true", help="Evaluate detection")
    parser.add_argument("--eval-recognition", action="store_true", help="Evaluate recognition")
    parser.add_argument("--threshold", type=float, default=0.6, help="Recognition threshold")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Initialize models
    print("Initializing models...")
    detector = FaceDetector(model_name="mtcnn", device="cpu")
    extractor = FeatureExtractor(model_name="facenet", device="cpu")
    matcher = FaceMatcher(use_faiss=True)
    
    results = {}
    
    # Evaluate detection
    if args.eval_detection:
        results["detection"] = evaluate_detection(detector, args.test)
    
    # Evaluate recognition
    if args.eval_recognition:
        results["recognition"] = evaluate_recognition(
            detector, extractor, matcher,
            args.gallery, args.test,
            threshold=args.threshold
        )
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
