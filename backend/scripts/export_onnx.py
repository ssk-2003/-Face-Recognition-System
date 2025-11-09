"""
Script to export PyTorch models to ONNX format for optimized CPU inference
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import onnx
import onnxruntime as ort
from pathlib import Path

from models.feature_extractor import FeatureExtractor, export_to_onnx
from utils.logger import log


def verify_onnx_model(onnx_path: str, input_shape: tuple = (1, 3, 160, 160)):
    """
    Verify exported ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape
    """
    print(f"\nVerifying ONNX model: {onnx_path}")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # Create test input
    test_input = torch.randn(*input_shape).numpy()
    
    # Test with ONNX Runtime
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    result = session.run([output_name], {input_name: test_input})
    print(f"✓ ONNX Runtime inference successful")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {result[0].shape}")
    
    return True


def benchmark_onnx_vs_pytorch(pytorch_model, onnx_path: str, num_runs: int = 100):
    """
    Compare PyTorch and ONNX inference speed
    
    Args:
        pytorch_model: PyTorch model
        onnx_path: Path to ONNX model
        num_runs: Number of benchmark runs
    """
    import time
    import numpy as np
    
    print(f"\nBenchmarking PyTorch vs ONNX (n={num_runs})...")
    
    # Test input
    input_shape = (1, 3, 160, 160)
    test_input = torch.randn(*input_shape)
    
    # Benchmark PyTorch
    pytorch_model.eval()
    pytorch_times = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = pytorch_model(test_input)
        
        # Benchmark
        for _ in range(num_runs):
            start = time.time()
            _ = pytorch_model(test_input)
            pytorch_times.append((time.time() - start) * 1000)
    
    # Benchmark ONNX
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    test_input_np = test_input.numpy()
    
    onnx_times = []
    
    # Warmup
    for _ in range(10):
        _ = session.run([output_name], {input_name: test_input_np})
    
    # Benchmark
    for _ in range(num_runs):
        start = time.time()
        _ = session.run([output_name], {input_name: test_input_np})
        onnx_times.append((time.time() - start) * 1000)
    
    # Results
    pytorch_times = np.array(pytorch_times)
    onnx_times = np.array(onnx_times)
    
    print("\nResults:")
    print(f"PyTorch:")
    print(f"  Mean: {pytorch_times.mean():.2f} ms")
    print(f"  Std: {pytorch_times.std():.2f} ms")
    print(f"  Throughput: {1000/pytorch_times.mean():.2f} FPS")
    
    print(f"\nONNX:")
    print(f"  Mean: {onnx_times.mean():.2f} ms")
    print(f"  Std: {onnx_times.std():.2f} ms")
    print(f"  Throughput: {1000/onnx_times.mean():.2f} FPS")
    
    speedup = pytorch_times.mean() / onnx_times.mean()
    print(f"\nSpeedup: {speedup:.2f}x")
    
    return {
        "pytorch_mean_ms": float(pytorch_times.mean()),
        "onnx_mean_ms": float(onnx_times.mean()),
        "speedup": float(speedup)
    }


def main():
    parser = argparse.ArgumentParser(description="Export models to ONNX")
    parser.add_argument("--model", type=str, default="facenet", choices=["facenet", "arcface"],
                       help="Model to export")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--verify", action="store_true", help="Verify exported model")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark ONNX vs PyTorch")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PyTorch model
    print(f"Loading {args.model} model...")
    extractor = FeatureExtractor(model_name=args.model, device="cpu")
    
    # Export to ONNX
    onnx_path = output_dir / f"{args.model}_embedding.onnx"
    print(f"\nExporting to ONNX: {onnx_path}")
    
    export_to_onnx(
        model=extractor.model,
        save_path=str(onnx_path),
        input_size=(1, 3, 160, 160)
    )
    
    print(f"✓ Model exported successfully")
    
    # Verify if requested
    if args.verify:
        verify_onnx_model(str(onnx_path))
    
    # Benchmark if requested
    if args.benchmark:
        benchmark_onnx_vs_pytorch(extractor.model, str(onnx_path))
    
    print(f"\n✓ Complete! ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()
