"""
Comprehensive CPU Benchmarking Script
Measures latency, throughput, and performance metrics for FRS
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import cv2
from pathlib import Path
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.detect_faces import FaceDetector
from src.extract_embeddings import FeatureExtractor
from src.match_faces import FaceMatcher
from src.utils import load_image
from src.config import settings


class PerformanceBenchmark:
    """Benchmark FRS components on CPU"""
    
    def __init__(self, test_images_dir: str = None):
        """Initialize benchmark with test images directory"""
        self.detector = FaceDetector()
        self.extractor = FeatureExtractor()
        self.matcher = FaceMatcher()
        
        # Find test images
        if test_images_dir:
            self.test_dir = Path(test_images_dir)
        else:
            self.test_dir = Path("data/test_images")
            
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "device": "CPU",
            "detection": {},
            "extraction": {},
            "matching": {},
            "end_to_end": {},
            "system_info": self._get_system_info()
        }
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": platform.python_version()
        }
    
    def _load_test_images(self) -> List[np.ndarray]:
        """Load test images"""
        images = []
        if not self.test_dir.exists():
            print(f"Warning: Test directory {self.test_dir} not found")
            print("Creating synthetic test images...")
            return self._create_synthetic_images()
        
        for img_path in self.test_dir.glob("*.jpg"):
            img = load_image(str(img_path))
            if img is not None:
                images.append(img)
        
        if not images:
            print("No test images found, creating synthetic images...")
            return self._create_synthetic_images()
            
        return images
    
    def _create_synthetic_images(self, count: int = 10) -> List[np.ndarray]:
        """Create synthetic test images"""
        images = []
        for i in range(count):
            # Create random image
            img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            images.append(img)
        return images
    
    def benchmark_detection(self, images: List[np.ndarray], iterations: int = 100) -> Dict:
        """Benchmark face detection"""
        print(f"\n{'='*60}")
        print("BENCHMARKING FACE DETECTION")
        print(f"{'='*60}")
        print(f"Images: {len(images)}, Iterations per image: {iterations}")
        
        latencies = []
        detections_count = []
        
        for img in images:
            for _ in range(iterations):
                start = time.perf_counter()
                boxes, _, confs = self.detector.detect(img, return_confidence=True)
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)  # Convert to ms
                detections_count.append(len(boxes))
        
        results = {
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_fps": 1000 / np.mean(latencies),
            "avg_faces_detected": np.mean(detections_count),
            "total_iterations": len(latencies)
        }
        
        self.results["detection"] = results
        self._print_results("Detection", results)
        return results
    
    def benchmark_extraction(self, images: List[np.ndarray], iterations: int = 100) -> Dict:
        """Benchmark feature extraction"""
        print(f"\n{'='*60}")
        print("BENCHMARKING FEATURE EXTRACTION")
        print(f"{'='*60}")
        
        # First detect faces
        face_crops = []
        for img in images:
            boxes, _, _ = self.detector.detect(img)
            if len(boxes) > 0:
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box)
                face = img[y1:y2, x1:x2]
                if face.size > 0:
                    face_crops.append(face)
        
        if not face_crops:
            print("No faces detected in test images for extraction benchmark")
            return {}
        
        print(f"Face crops: {len(face_crops)}, Iterations per face: {iterations}")
        
        latencies = []
        embedding_dims = []
        
        for face in face_crops:
            for _ in range(iterations):
                start = time.perf_counter()
                embedding = self.extractor.extract(face)
                end = time.perf_counter()
                
                latencies.append((end - start) * 1000)
                embedding_dims.append(len(embedding))
        
        results = {
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "std_latency_ms": np.std(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_fps": 1000 / np.mean(latencies),
            "embedding_dimension": int(np.mean(embedding_dims)),
            "total_iterations": len(latencies)
        }
        
        self.results["extraction"] = results
        self._print_results("Feature Extraction", results)
        return results
    
    def benchmark_matching(self, gallery_size: int = 100, iterations: int = 1000) -> Dict:
        """Benchmark face matching against gallery"""
        print(f"\n{'='*60}")
        print("BENCHMARKING FACE MATCHING")
        print(f"{'='*60}")
        print(f"Gallery size: {gallery_size}, Query iterations: {iterations}")
        
        # Create mock gallery
        for i in range(gallery_size):
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            self.matcher.add_embedding(embedding, i, f"Person_{i}")
        
        latencies = []
        
        for _ in range(iterations):
            query_embedding = np.random.randn(512).astype(np.float32)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            start = time.perf_counter()
            matches = self.matcher.match(query_embedding, threshold=0.6, top_k=5)
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)
        
        results = {
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "throughput_qps": 1000 / np.mean(latencies),
            "gallery_size": gallery_size,
            "total_iterations": iterations
        }
        
        self.results["matching"] = results
        self._print_results("Matching", results)
        return results
    
    def benchmark_end_to_end(self, images: List[np.ndarray], iterations: int = 50) -> Dict:
        """Benchmark complete pipeline"""
        print(f"\n{'='*60}")
        print("BENCHMARKING END-TO-END PIPELINE")
        print(f"{'='*60}")
        print(f"Images: {len(images)}, Iterations: {iterations}")
        
        # Setup gallery
        gallery_size = 50
        for i in range(gallery_size):
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            self.matcher.add_embedding(embedding, i, f"Person_{i}")
        
        latencies = {
            "total": [],
            "detection": [],
            "extraction": [],
            "matching": []
        }
        
        successful_recognitions = 0
        total_faces = 0
        
        for img in images:
            for _ in range(iterations):
                # Total pipeline
                t_start = time.perf_counter()
                
                # Detection
                t_det_start = time.perf_counter()
                boxes, _, _ = self.detector.detect(img)
                t_det_end = time.perf_counter()
                latencies["detection"].append((t_det_end - t_det_start) * 1000)
                
                if len(boxes) == 0:
                    continue
                
                total_faces += len(boxes)
                
                # Extraction and matching for first face
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box)
                face = img[y1:y2, x1:x2]
                
                if face.size > 0:
                    # Extraction
                    t_ext_start = time.perf_counter()
                    embedding = self.extractor.extract(face)
                    t_ext_end = time.perf_counter()
                    latencies["extraction"].append((t_ext_end - t_ext_start) * 1000)
                    
                    # Matching
                    t_match_start = time.perf_counter()
                    matches = self.matcher.match(embedding, threshold=0.6, top_k=1)
                    t_match_end = time.perf_counter()
                    latencies["matching"].append((t_match_end - t_match_start) * 1000)
                    
                    if matches:
                        successful_recognitions += 1
                
                t_end = time.perf_counter()
                latencies["total"].append((t_end - t_start) * 1000)
        
        results = {
            "total_mean_latency_ms": np.mean(latencies["total"]),
            "total_p95_latency_ms": np.percentile(latencies["total"], 95),
            "detection_mean_ms": np.mean(latencies["detection"]) if latencies["detection"] else 0,
            "extraction_mean_ms": np.mean(latencies["extraction"]) if latencies["extraction"] else 0,
            "matching_mean_ms": np.mean(latencies["matching"]) if latencies["matching"] else 0,
            "throughput_fps": 1000 / np.mean(latencies["total"]) if latencies["total"] else 0,
            "recognition_rate": (successful_recognitions / total_faces * 100) if total_faces > 0 else 0,
            "total_faces_processed": total_faces
        }
        
        self.results["end_to_end"] = results
        self._print_results("End-to-End", results)
        return results
    
    def _print_results(self, name: str, results: Dict):
        """Pretty print results"""
        print(f"\n{name} Results:")
        print("-" * 40)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    def generate_report(self, output_dir: str = "reports") -> str:
        """Generate comprehensive report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = output_path / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate plots
        self._generate_plots(output_path, timestamp)
        
        # Generate markdown report
        md_file = output_path / f"benchmark_report_{timestamp}.md"
        self._generate_markdown_report(md_file)
        
        print(f"\n{'='*60}")
        print(f"Reports generated:")
        print(f"  JSON: {json_file}")
        print(f"  Markdown: {md_file}")
        print(f"  Plots: {output_path}")
        print(f"{'='*60}")
        
        return str(md_file)
    
    def _generate_plots(self, output_dir: Path, timestamp: str):
        """Generate visualization plots"""
        sns.set_style("whitegrid")
        
        # Latency comparison
        if all(k in self.results for k in ["detection", "extraction", "matching"]):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            components = ["Detection", "Extraction", "Matching"]
            latencies = [
                self.results["detection"].get("mean_latency_ms", 0),
                self.results["extraction"].get("mean_latency_ms", 0),
                self.results["matching"].get("mean_latency_ms", 0)
            ]
            
            bars = ax.bar(components, latencies, color=['#3498db', '#e74c3c', '#2ecc71'])
            ax.set_ylabel('Latency (ms)', fontsize=12)
            ax.set_title('Component Latency Comparison (CPU)', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}ms',
                       ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"latency_comparison_{timestamp}.png", dpi=300)
            plt.close()
        
        # Throughput chart
        if "end_to_end" in self.results and "throughput_fps" in self.results["end_to_end"]:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            fps = self.results["end_to_end"]["throughput_fps"]
            ax.barh(['End-to-End'], [fps], color='#9b59b6')
            ax.set_xlabel('Throughput (FPS)', fontsize=12)
            ax.set_title('System Throughput (CPU)', fontsize=14, fontweight='bold')
            ax.text(fps + 0.1, 0, f'{fps:.2f} FPS', va='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / f"throughput_{timestamp}.png", dpi=300)
            plt.close()
    
    def _generate_markdown_report(self, output_file: Path):
        """Generate markdown report"""
        with open(output_file, 'w') as f:
            f.write("# Face Recognition System - Performance Benchmark Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")
            
            # System Info
            f.write("## System Information\n\n")
            for key, value in self.results['system_info'].items():
                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
            f.write("\n")
            
            # Detection Results
            if "detection" in self.results and self.results["detection"]:
                f.write("## Face Detection Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in self.results["detection"].items():
                    formatted_value = f"{value:.3f}" if isinstance(value, float) else value
                    f.write(f"| {key.replace('_', ' ').title()} | {formatted_value} |\n")
                f.write("\n")
            
            # Extraction Results
            if "extraction" in self.results and self.results["extraction"]:
                f.write("## Feature Extraction Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in self.results["extraction"].items():
                    formatted_value = f"{value:.3f}" if isinstance(value, float) else value
                    f.write(f"| {key.replace('_', ' ').title()} | {formatted_value} |\n")
                f.write("\n")
            
            # Matching Results
            if "matching" in self.results and self.results["matching"]:
                f.write("## Face Matching Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in self.results["matching"].items():
                    formatted_value = f"{value:.3f}" if isinstance(value, float) else value
                    f.write(f"| {key.replace('_', ' ').title()} | {formatted_value} |\n")
                f.write("\n")
            
            # End-to-End Results
            if "end_to_end" in self.results and self.results["end_to_end"]:
                f.write("## End-to-End Pipeline Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in self.results["end_to_end"].items():
                    formatted_value = f"{value:.3f}" if isinstance(value, float) else value
                    f.write(f"| {key.replace('_', ' ').title()} | {formatted_value} |\n")
                f.write("\n")
            
            # Summary
            f.write("## Summary\n\n")
            if "detection" in self.results and self.results["detection"]:
                det_latency = self.results["detection"].get("mean_latency_ms", 0)
                det_fps = self.results["detection"].get("throughput_fps", 0)
                f.write(f"- **Detection:** {det_latency:.2f}ms average latency, {det_fps:.2f} FPS\n")
            
            if "extraction" in self.results and self.results["extraction"]:
                ext_latency = self.results["extraction"].get("mean_latency_ms", 0)
                f.write(f"- **Extraction:** {ext_latency:.2f}ms average latency\n")
            
            if "matching" in self.results and self.results["matching"]:
                match_latency = self.results["matching"].get("mean_latency_ms", 0)
                f.write(f"- **Matching:** {match_latency:.2f}ms average latency\n")
            
            if "end_to_end" in self.results and self.results["end_to_end"]:
                e2e_latency = self.results["end_to_end"].get("total_mean_latency_ms", 0)
                e2e_fps = self.results["end_to_end"].get("throughput_fps", 0)
                f.write(f"- **End-to-End Pipeline:** {e2e_latency:.2f}ms average latency, {e2e_fps:.2f} FPS\n")
            
            f.write("\n---\n")
            f.write("*Report generated by FRS Benchmark Suite*\n")


def main():
    """Run comprehensive benchmark"""
    print("\n" + "="*60)
    print("FACE RECOGNITION SYSTEM - CPU BENCHMARK")
    print("="*60)
    
    # Initialize benchmark
    benchmark = PerformanceBenchmark()
    
    # Load test images
    print("\nLoading test images...")
    images = benchmark._load_test_images()
    print(f"Loaded {len(images)} test images")
    
    # Run benchmarks
    benchmark.benchmark_detection(images, iterations=50)
    benchmark.benchmark_extraction(images, iterations=50)
    benchmark.benchmark_matching(gallery_size=100, iterations=500)
    benchmark.benchmark_end_to_end(images, iterations=20)
    
    # Generate report
    benchmark.generate_report()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
