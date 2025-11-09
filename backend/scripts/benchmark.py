"""
Benchmark script for Face Recognition Service
Measures detection, extraction, and matching performance
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from models.face_detector import FaceDetector
from models.feature_extractor import FeatureExtractor
from models.matcher import FaceMatcher
from utils.image_processing import load_image, crop_face
from utils.logger import log


def benchmark_detection(detector: FaceDetector, images: list, num_runs: int = 100):
    """Benchmark face detection"""
    print("\n" + "="*50)
    print("Benchmarking Face Detection")
    print("="*50)
    
    times = []
    detections = []
    
    for _ in tqdm(range(num_runs), desc="Detection"):
        img = images[np.random.randint(len(images))]
        
        start = time.time()
        boxes, _, _ = detector.detect(img, return_landmarks=True, return_confidence=True)
        elapsed = (time.time() - start) * 1000
        
        times.append(elapsed)
        detections.append(len(boxes))
    
    times = np.array(times)
    
    print(f"\nResults (n={num_runs}):")
    print(f"  Mean time: {times.mean():.2f} ms")
    print(f"  Median time: {np.median(times):.2f} ms")
    print(f"  Min time: {times.min():.2f} ms")
    print(f"  Max time: {times.max():.2f} ms")
    print(f"  Std dev: {times.std():.2f} ms")
    print(f"  Throughput: {1000/times.mean():.2f} FPS")
    print(f"  Avg faces per image: {np.mean(detections):.2f}")
    
    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "std_ms": float(times.std()),
        "fps": float(1000/times.mean()),
        "avg_faces": float(np.mean(detections))
    }


def benchmark_extraction(extractor: FeatureExtractor, faces: list, num_runs: int = 100):
    """Benchmark feature extraction"""
    print("\n" + "="*50)
    print("Benchmarking Feature Extraction")
    print("="*50)
    
    times = []
    
    for _ in tqdm(range(num_runs), desc="Extraction"):
        face = faces[np.random.randint(len(faces))]
        
        start = time.time()
        embedding = extractor.extract(face)
        elapsed = (time.time() - start) * 1000
        
        times.append(elapsed)
    
    times = np.array(times)
    
    print(f"\nResults (n={num_runs}):")
    print(f"  Mean time: {times.mean():.2f} ms")
    print(f"  Median time: {np.median(times):.2f} ms")
    print(f"  Min time: {times.min():.2f} ms")
    print(f"  Max time: {times.max():.2f} ms")
    print(f"  Std dev: {times.std():.2f} ms")
    print(f"  Throughput: {1000/times.mean():.2f} FPS")
    
    return {
        "mean_ms": float(times.mean()),
        "median_ms": float(np.median(times)),
        "min_ms": float(times.min()),
        "max_ms": float(times.max()),
        "std_ms": float(times.std()),
        "fps": float(1000/times.mean())
    }


def benchmark_matching(matcher: FaceMatcher, embeddings: list, gallery_sizes: list = [10, 50, 100, 500]):
    """Benchmark matching with different gallery sizes"""
    print("\n" + "="*50)
    print("Benchmarking Matching")
    print("="*50)
    
    results = {}
    
    for gallery_size in gallery_sizes:
        if gallery_size > len(embeddings):
            continue
        
        # Create gallery
        matcher.clear()
        for i, emb in enumerate(embeddings[:gallery_size]):
            matcher.add_embedding(emb, i, f"person_{i}")
        
        # Benchmark
        times = []
        for _ in tqdm(range(100), desc=f"Gallery size {gallery_size}"):
            query = embeddings[np.random.randint(len(embeddings))]
            
            start = time.time()
            matches = matcher.match(query, threshold=0.5, top_k=5)
            elapsed = (time.time() - start) * 1000
            
            times.append(elapsed)
        
        times = np.array(times)
        
        results[gallery_size] = {
            "mean_ms": float(times.mean()),
            "median_ms": float(np.median(times)),
            "min_ms": float(times.min()),
            "max_ms": float(times.max()),
            "fps": float(1000/times.mean())
        }
        
        print(f"\nGallery size: {gallery_size}")
        print(f"  Mean time: {times.mean():.2f} ms")
        print(f"  Median time: {np.median(times):.2f} ms")
        print(f"  Throughput: {1000/times.mean():.2f} FPS")
    
    return results


def benchmark_end_to_end(detector, extractor, matcher, images: list, num_runs: int = 50):
    """Benchmark complete pipeline"""
    print("\n" + "="*50)
    print("Benchmarking End-to-End Pipeline")
    print("="*50)
    
    times = {
        "detection": [],
        "extraction": [],
        "matching": [],
        "total": []
    }
    
    for _ in tqdm(range(num_runs), desc="End-to-End"):
        img = images[np.random.randint(len(images))]
        
        # Detection
        start = time.time()
        boxes, _, _ = detector.detect(img, return_landmarks=True, return_confidence=True)
        det_time = (time.time() - start) * 1000
        times["detection"].append(det_time)
        
        if len(boxes) == 0:
            continue
        
        # Extraction
        face = crop_face(img, boxes[0])
        start = time.time()
        embedding = extractor.extract(face)
        ext_time = (time.time() - start) * 1000
        times["extraction"].append(ext_time)
        
        # Matching
        start = time.time()
        matches = matcher.match(embedding)
        match_time = (time.time() - start) * 1000
        times["matching"].append(match_time)
        
        times["total"].append(det_time + ext_time + match_time)
    
    results = {}
    for key, vals in times.items():
        if len(vals) > 0:
            vals = np.array(vals)
            results[key] = {
                "mean_ms": float(vals.mean()),
                "median_ms": float(np.median(vals)),
                "std_ms": float(vals.std())
            }
            print(f"\n{key.capitalize()}:")
            print(f"  Mean: {vals.mean():.2f} ms")
            print(f"  Median: {np.median(vals):.2f} ms")
    
    if len(times["total"]) > 0:
        total = np.array(times["total"])
        print(f"\nTotal pipeline:")
        print(f"  Mean: {total.mean():.2f} ms")
        print(f"  Throughput: {1000/total.mean():.2f} FPS")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Face Recognition Service")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--num-runs", type=int, default=100, help="Number of runs per benchmark")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    
    args = parser.parse_args()
    
    # Load test images
    print("Loading test images...")
    data_path = Path(args.data_dir)
    image_files = list(data_path.rglob("*.jpg")) + list(data_path.rglob("*.png"))
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    images = [load_image(str(f)) for f in image_files[:100]]  # Limit to 100 images
    print(f"Loaded {len(images)} images")
    
    # Initialize models
    print("\nInitializing models...")
    detector = FaceDetector(model_name="mtcnn", device="cpu")
    extractor = FeatureExtractor(model_name="facenet", device="cpu")
    matcher = FaceMatcher(use_faiss=True)
    
    # Detect and extract faces for benchmarking
    print("\nPreparing test faces...")
    faces = []
    embeddings = []
    for img in tqdm(images[:50]):
        boxes, _, _ = detector.detect(img)
        if len(boxes) > 0:
            face = crop_face(img, boxes[0])
            faces.append(face)
            embedding = extractor.extract(face)
            embeddings.append(embedding)
    
    print(f"Prepared {len(faces)} faces")
    
    # Initialize gallery
    for i, emb in enumerate(embeddings[:100]):
        matcher.add_embedding(emb, i, f"person_{i}")
    
    # Run benchmarks
    results = {
        "system": {
            "device": "cpu",
            "num_test_images": len(images),
            "num_test_faces": len(faces)
        }
    }
    
    results["detection"] = benchmark_detection(detector, images, args.num_runs)
    results["extraction"] = benchmark_extraction(extractor, faces, args.num_runs)
    results["matching"] = benchmark_matching(matcher, embeddings)
    results["end_to_end"] = benchmark_end_to_end(detector, extractor, matcher, images, args.num_runs)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
