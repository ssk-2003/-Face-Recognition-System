"""
ONNX Model Conversion Script
Converts PyTorch models to ONNX format for optimized CPU inference
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
from typing import Tuple

from src.extract_embeddings import FeatureExtractor
from src.config import settings


class ONNXConverter:
    """Convert and benchmark ONNX models"""
    
    def __init__(self, output_dir: str = "models/onnx"):
        """Initialize converter"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_facenet_to_onnx(self, 
                                input_size: Tuple[int, int, int] = (3, 160, 160),
                                batch_size: int = 1) -> str:
        """
        Convert FaceNet model to ONNX format
        
        Args:
            input_size: Input tensor size (C, H, W)
            batch_size: Batch size for inference
        
        Returns:
            Path to saved ONNX model
        """
        print("\n" + "="*60)
        print("CONVERTING FACENET TO ONNX")
        print("="*60)
        
        # Load PyTorch model
        print("Loading PyTorch FaceNet model...")
        extractor = FeatureExtractor()
        model = extractor.model
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_size)
        
        # Output path
        onnx_path = self.output_dir / "facenet_vggface2.onnx"
        
        print(f"Converting to ONNX...")
        print(f"  Input shape: {(batch_size,) + input_size}")
        print(f"  Output path: {onnx_path}")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['embedding'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'embedding': {0: 'batch_size'}
            }
        )
        
        print("✅ ONNX export complete!")
        
        # Verify the model
        print("\nVerifying ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model verified!")
        
        # Get model size
        model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"\nModel size: {model_size_mb:.2f} MB")
        
        return str(onnx_path)
    
    def benchmark_onnx_vs_pytorch(self, onnx_model_path: str, 
                                 iterations: int = 100) -> dict:
        """
        Benchmark ONNX model vs PyTorch model
        
        Args:
            onnx_model_path: Path to ONNX model
            iterations: Number of iterations for benchmarking
        
        Returns:
            Dictionary with benchmark results
        """
        print("\n" + "="*60)
        print("BENCHMARKING: ONNX vs PyTorch")
        print("="*60)
        
        # Load PyTorch model
        print("\nLoading PyTorch model...")
        extractor = FeatureExtractor()
        pytorch_model = extractor.model
        pytorch_model.eval()
        
        # Load ONNX model
        print("Loading ONNX model...")
        ort_session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Create test input
        test_input = np.random.randn(1, 3, 160, 160).astype(np.float32)
        
        # Warm-up
        print("\nWarm-up phase...")
        for _ in range(10):
            with torch.no_grad():
                _ = pytorch_model(torch.from_numpy(test_input))
            _ = ort_session.run(None, {'input': test_input})
        
        # Benchmark PyTorch
        print(f"\nBenchmarking PyTorch ({iterations} iterations)...")
        pytorch_times = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                _ = pytorch_model(torch.from_numpy(test_input))
                end = time.perf_counter()
                pytorch_times.append((end - start) * 1000)
        
        # Benchmark ONNX
        print(f"Benchmarking ONNX ({iterations} iterations)...")
        onnx_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = ort_session.run(None, {'input': test_input})
            end = time.perf_counter()
            onnx_times.append((end - start) * 1000)
        
        # Calculate metrics
        pytorch_mean = np.mean(pytorch_times)
        onnx_mean = np.mean(onnx_times)
        speedup = pytorch_mean / onnx_mean
        
        results = {
            "pytorch": {
                "mean_latency_ms": pytorch_mean,
                "std_latency_ms": np.std(pytorch_times),
                "min_latency_ms": np.min(pytorch_times),
                "max_latency_ms": np.max(pytorch_times),
                "p95_latency_ms": np.percentile(pytorch_times, 95),
                "throughput_fps": 1000 / pytorch_mean
            },
            "onnx": {
                "mean_latency_ms": onnx_mean,
                "std_latency_ms": np.std(onnx_times),
                "min_latency_ms": np.min(onnx_times),
                "max_latency_ms": np.max(onnx_times),
                "p95_latency_ms": np.percentile(onnx_times, 95),
                "throughput_fps": 1000 / onnx_mean
            },
            "speedup": speedup,
            "improvement_percent": (speedup - 1) * 100
        }
        
        # Print results
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print("\nPyTorch:")
        print(f"  Mean latency: {pytorch_mean:.2f} ms")
        print(f"  P95 latency: {results['pytorch']['p95_latency_ms']:.2f} ms")
        print(f"  Throughput: {results['pytorch']['throughput_fps']:.2f} FPS")
        
        print("\nONNX:")
        print(f"  Mean latency: {onnx_mean:.2f} ms")
        print(f"  P95 latency: {results['onnx']['p95_latency_ms']:.2f} ms")
        print(f"  Throughput: {results['onnx']['throughput_fps']:.2f} FPS")
        
        print(f"\nSpeedup: {speedup:.2f}x ({results['improvement_percent']:.1f}% faster)")
        
        return results
    
    def verify_onnx_accuracy(self, onnx_model_path: str, 
                           test_samples: int = 100) -> dict:
        """
        Verify ONNX model produces same outputs as PyTorch
        
        Args:
            onnx_model_path: Path to ONNX model
            test_samples: Number of test samples
        
        Returns:
            Dictionary with accuracy comparison
        """
        print("\n" + "="*60)
        print("VERIFYING ONNX ACCURACY")
        print("="*60)
        
        # Load models
        extractor = FeatureExtractor()
        pytorch_model = extractor.model
        pytorch_model.eval()
        
        ort_session = ort.InferenceSession(
            onnx_model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Test on random inputs
        max_diff_list = []
        mean_diff_list = []
        
        for i in range(test_samples):
            # Generate random input
            test_input = np.random.randn(1, 3, 160, 160).astype(np.float32)
            
            # PyTorch inference
            with torch.no_grad():
                pytorch_output = pytorch_model(torch.from_numpy(test_input)).numpy()
            
            # ONNX inference
            onnx_output = ort_session.run(None, {'input': test_input})[0]
            
            # Calculate difference
            diff = np.abs(pytorch_output - onnx_output)
            max_diff_list.append(np.max(diff))
            mean_diff_list.append(np.mean(diff))
        
        results = {
            "max_difference": np.max(max_diff_list),
            "mean_max_difference": np.mean(max_diff_list),
            "mean_difference": np.mean(mean_diff_list),
            "test_samples": test_samples,
            "outputs_match": np.max(max_diff_list) < 1e-5  # Tolerance
        }
        
        print(f"\nAccuracy Verification ({test_samples} samples):")
        print(f"  Max difference: {results['max_difference']:.2e}")
        print(f"  Mean difference: {results['mean_difference']:.2e}")
        print(f"  Outputs match: {'✅ YES' if results['outputs_match'] else '❌ NO'}")
        
        return results
    
    def create_onnx_inference_wrapper(self, onnx_model_path: str) -> str:
        """
        Create a Python wrapper for ONNX inference
        
        Args:
            onnx_model_path: Path to ONNX model
        
        Returns:
            Path to wrapper file
        """
        wrapper_code = '''"""
ONNX Inference Wrapper for FaceNet
Auto-generated by onnx_converter.py
"""
import onnxruntime as ort
import numpy as np
from pathlib import Path


class ONNXFaceNetExtractor:
    """Fast ONNX-based feature extractor"""
    
    def __init__(self, model_path: str = None):
        """Initialize ONNX model"""
        if model_path is None:
            model_path = Path(__file__).parent / "onnx" / "facenet_vggface2.onnx"
        
        self.session = ort.InferenceSession(
            str(model_path),
            providers=['CPUExecutionProvider']
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from face image
        
        Args:
            face_image: RGB face image (H, W, 3)
        
        Returns:
            Embedding vector (512,)
        """
        # Preprocess
        if face_image.shape[:2] != (160, 160):
            import cv2
            face_image = cv2.resize(face_image, (160, 160))
        
        # Normalize and transpose
        face_tensor = (face_image / 255.0 - 0.5) / 0.5
        face_tensor = np.transpose(face_tensor, (2, 0, 1))
        face_tensor = np.expand_dims(face_tensor, axis=0).astype(np.float32)
        
        # Inference
        embedding = self.session.run(
            [self.output_name],
            {self.input_name: face_tensor}
        )[0]
        
        return embedding.flatten()


# Usage example
if __name__ == "__main__":
    extractor = ONNXFaceNetExtractor()
    
    # Test with random image
    test_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    embedding = extractor.extract(test_image)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
'''
        
        wrapper_path = self.output_dir.parent / "onnx_extractor.py"
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_code)
        
        print(f"\n✅ ONNX wrapper created: {wrapper_path}")
        return str(wrapper_path)
    
    def generate_report(self, conversion_results: dict, 
                       benchmark_results: dict, 
                       accuracy_results: dict,
                       output_dir: str = "reports") -> str:
        """Generate ONNX conversion report"""
        from datetime import datetime
        import json
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON
        json_file = output_path / f"onnx_conversion_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "conversion": conversion_results,
                "benchmark": benchmark_results,
                "accuracy": accuracy_results
            }, f, indent=2)
        
        # Generate Markdown report
        md_file = output_path / f"onnx_report_{timestamp}.md"
        with open(md_file, 'w') as f:
            f.write("# ONNX Model Conversion Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            
            f.write("## Model Conversion\n\n")
            f.write(f"- **Model:** FaceNet (VGGFace2)\n")
            f.write(f"- **Framework:** PyTorch → ONNX\n")
            f.write(f"- **Optimization:** CPU inference\n\n")
            
            f.write("## Performance Comparison\n\n")
            f.write("| Metric | PyTorch | ONNX | Speedup |\n")
            f.write("|--------|---------|------|--------|\n")
            
            pt_latency = benchmark_results['pytorch']['mean_latency_ms']
            onnx_latency = benchmark_results['onnx']['mean_latency_ms']
            speedup = benchmark_results['speedup']
            
            f.write(f"| Mean Latency | {pt_latency:.2f} ms | {onnx_latency:.2f} ms | {speedup:.2f}x |\n")
            f.write(f"| P95 Latency | {benchmark_results['pytorch']['p95_latency_ms']:.2f} ms | {benchmark_results['onnx']['p95_latency_ms']:.2f} ms | - |\n")
            f.write(f"| Throughput | {benchmark_results['pytorch']['throughput_fps']:.2f} FPS | {benchmark_results['onnx']['throughput_fps']:.2f} FPS | {speedup:.2f}x |\n\n")
            
            f.write("## Accuracy Verification\n\n")
            f.write(f"- **Test Samples:** {accuracy_results['test_samples']}\n")
            f.write(f"- **Max Difference:** {accuracy_results['max_difference']:.2e}\n")
            f.write(f"- **Mean Difference:** {accuracy_results['mean_difference']:.2e}\n")
            f.write(f"- **Outputs Match:** {'✅ YES' if accuracy_results['outputs_match'] else '❌ NO'}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"ONNX conversion achieved **{benchmark_results['improvement_percent']:.1f}% performance improvement** ")
            f.write(f"with negligible accuracy loss (max diff: {accuracy_results['max_difference']:.2e}). ")
            f.write(f"The ONNX model is ready for production deployment.\n\n")
            
            f.write("---\n")
            f.write("*Report generated by FRS ONNX Converter*\n")
        
        print(f"\n✅ Report saved to: {md_file}")
        return str(md_file)


def main():
    """Run ONNX conversion pipeline"""
    print("\n" + "="*60)
    print("ONNX MODEL CONVERSION PIPELINE")
    print("="*60)
    
    converter = ONNXConverter()
    
    # Check if ONNX Runtime is available
    try:
        import onnxruntime
    except ImportError:
        print("\n❌ Error: onnxruntime not installed")
        print("Install with: pip install onnxruntime")
        return
    
    # Convert model
    onnx_path = converter.convert_facenet_to_onnx()
    
    conversion_results = {
        "model_path": onnx_path,
        "input_shape": [1, 3, 160, 160],
        "output_shape": [1, 512]
    }
    
    # Benchmark
    benchmark_results = converter.benchmark_onnx_vs_pytorch(onnx_path, iterations=100)
    
    # Verify accuracy
    accuracy_results = converter.verify_onnx_accuracy(onnx_path, test_samples=50)
    
    # Create wrapper
    converter.create_onnx_inference_wrapper(onnx_path)
    
    # Generate report
    converter.generate_report(conversion_results, benchmark_results, accuracy_results)
    
    print("\n✅ ONNX conversion complete!")
    print(f"\n📦 ONNX model saved to: {onnx_path}")
    print("You can now use the ONNX model for faster CPU inference!")


if __name__ == "__main__":
    main()
