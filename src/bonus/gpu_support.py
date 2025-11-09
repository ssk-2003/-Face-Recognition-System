"""
Multi-GPU Support Module
Enable distributed inference across multiple GPUs
"""

import torch
import torch.nn as nn
from typing import List, Optional
import os

class MultiGPUManager:
    """
    Manage multi-GPU inference for face recognition models
    """
    
    def __init__(self, model: nn.Module, gpu_ids: Optional[List[int]] = None):
        """
        Initialize multi-GPU manager
        
        Args:
            model: PyTorch model
            gpu_ids: List of GPU IDs to use (None = all available)
        """
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = torch.cuda.device_count()
        
        # Select GPUs
        if gpu_ids is None:
            self.gpu_ids = list(range(self.gpu_count))
        else:
            self.gpu_ids = gpu_ids
        
        self.distributed_model = None
        self._setup_multi_gpu()
    
    def _setup_multi_gpu(self):
        """Setup multi-GPU configuration"""
        if self.gpu_count > 1 and len(self.gpu_ids) > 1:
            print(f"Using {len(self.gpu_ids)} GPUs: {self.gpu_ids}")
            
            # DataParallel for simple multi-GPU
            self.distributed_model = nn.DataParallel(
                self.model,
                device_ids=self.gpu_ids
            )
            self.distributed_model.to(f'cuda:{self.gpu_ids[0]}')
            
        elif self.gpu_count >= 1:
            print(f"Using single GPU: {self.gpu_ids[0] if self.gpu_ids else 0}")
            self.model.to(self.device)
            self.distributed_model = self.model
        else:
            print("No GPU available, using CPU")
            self.distributed_model = self.model
    
    def get_model(self) -> nn.Module:
        """Get the distributed model"""
        return self.distributed_model
    
    def get_device(self) -> torch.device:
        """Get primary device"""
        return self.device
    
    @staticmethod
    def get_gpu_info() -> dict:
        """Get GPU information"""
        info = {
            'gpu_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count(),
            'gpus': []
        }
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory / 1e9,
                    'memory_allocated': torch.cuda.memory_allocated(i) / 1e9,
                    'memory_cached': torch.cuda.memory_reserved(i) / 1e9,
                }
                info['gpus'].append(gpu_info)
        
        return info
    
    @staticmethod
    def set_gpu_memory_growth(enable: bool = True):
        """Enable/disable GPU memory growth"""
        if torch.cuda.is_available():
            if enable:
                # PyTorch automatically manages memory growth
                torch.cuda.empty_cache()
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class DistributedInference:
    """
    Distributed inference for batch processing
    """
    
    def __init__(self, model: nn.Module, batch_size: int = 32, gpu_ids: Optional[List[int]] = None):
        """
        Initialize distributed inference
        
        Args:
            model: Model for inference
            batch_size: Batch size per GPU
            gpu_ids: GPUs to use
        """
        self.gpu_manager = MultiGPUManager(model, gpu_ids)
        self.model = self.gpu_manager.get_model()
        self.batch_size = batch_size
        self.device = self.gpu_manager.get_device()
    
    def predict_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict on a batch of inputs
        
        Args:
            inputs: Input tensor (B x C x H x W)
            
        Returns:
            Output tensor
        """
        self.model.eval()
        
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            outputs = self.model(inputs)
        
        return outputs
    
    def predict_large_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Predict on large batch by splitting into smaller batches
        
        Args:
            inputs: Input tensor
            
        Returns:
            Concatenated outputs
        """
        all_outputs = []
        
        for i in range(0, len(inputs), self.batch_size):
            batch = inputs[i:i + self.batch_size]
            outputs = self.predict_batch(batch)
            all_outputs.append(outputs.cpu())
        
        return torch.cat(all_outputs, dim=0)

def setup_distributed_training(rank: int, world_size: int):
    """
    Setup for distributed training (DDP)
    
    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    torch.distributed.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def cleanup_distributed():
    """Cleanup distributed training"""
    torch.distributed.destroy_process_group()

# Utility functions
def get_optimal_batch_size(model: nn.Module, input_shape: tuple, device: str = 'cuda') -> int:
    """
    Find optimal batch size for given model and input shape
    
    Args:
        model: PyTorch model
        input_shape: Input shape (C, H, W)
        device: Device to test on
        
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return 8  # Conservative for CPU
    
    model.eval()
    model.to(device)
    
    batch_size = 1
    max_batch_size = 128
    
    try:
        while batch_size <= max_batch_size:
            try:
                # Test with dummy input
                dummy_input = torch.randn(batch_size, *input_shape).to(device)
                
                with torch.no_grad():
                    _ = model(dummy_input)
                
                torch.cuda.synchronize()
                batch_size *= 2
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    torch.cuda.empty_cache()
                    return batch_size // 2
                else:
                    raise e
        
        return max_batch_size
        
    finally:
        torch.cuda.empty_cache()

# Example usage
if __name__ == "__main__":
    print("GPU Information:")
    print("-" * 50)
    
    info = MultiGPUManager.get_gpu_info()
    print(f"GPU Available: {info['gpu_available']}")
    print(f"Number of GPUs: {info['gpu_count']}")
    
    for gpu in info['gpus']:
        print(f"\nGPU {gpu['id']}: {gpu['name']}")
        print(f"  Total Memory: {gpu['memory_total']:.2f} GB")
        print(f"  Allocated: {gpu['memory_allocated']:.2f} GB")
        print(f"  Cached: {gpu['memory_cached']:.2f} GB")
    
    # Test with dummy model
    if torch.cuda.is_available():
        print("\nTesting Multi-GPU Setup:")
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        
        manager = MultiGPUManager(model)
        print(f"Model device: {manager.get_device()}")
        
        # Test optimal batch size
        optimal_bs = get_optimal_batch_size(model, (3, 224, 224))
        print(f"Optimal batch size: {optimal_bs}")
