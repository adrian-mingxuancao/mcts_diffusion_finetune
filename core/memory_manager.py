"""
Comprehensive Memory Management System for DPLM-2 and ESMFold
Allows all models to work together efficiently on the same GPU.
"""

import torch
import gc
import psutil
import os
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class GPUMemoryManager:
    """Comprehensive GPU memory manager for multiple large models"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model_registry: Dict[str, Any] = {}
        self.model_memory_usage: Dict[str, float] = {}
        self.current_loaded_models: List[str] = []
        
        # Model memory estimates (in GB)
        self.model_memory_estimates = {
            "dplm2_150m": 0.6,   # ~600MB
            "dplm2_650m": 1.3,   # ~1.3GB  
            "dplm2_3b": 6.0,     # ~6GB
            "esmfold": 4.0,      # ~4GB
        }
        
        # GPU memory thresholds
        self.gpu_memory_threshold = 0.85  # Use 85% of GPU memory max
        self.emergency_threshold = 0.95   # Emergency cleanup at 95%
        
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "free": 0, "usage_percent": 0}
            
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3   # GB
        free = total - allocated
        
        return {
            "total": total,
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "usage_percent": allocated / total
        }
    
    def can_load_model(self, model_name: str) -> bool:
        """Check if we can load a model without OOM"""
        memory_info = self.get_gpu_memory_info()
        required_memory = self.model_memory_estimates.get(model_name, 2.0)
        
        # Check if we have enough free memory
        if memory_info["free"] < required_memory:
            return False
            
        # Check if loading would exceed threshold
        projected_usage = (memory_info["allocated"] + required_memory) / memory_info["total"]
        return projected_usage < self.gpu_memory_threshold
    
    def emergency_cleanup(self):
        """Emergency memory cleanup"""
        print("🚨 Emergency GPU memory cleanup...")
        
        # Clear all caches
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Unload all models except the most recently used
        if len(self.current_loaded_models) > 1:
            models_to_unload = self.current_loaded_models[:-1]  # Keep the last one
            for model_name in models_to_unload:
                self.unload_model(model_name)
        
        print(f"✅ Emergency cleanup completed. Memory usage: {self.get_gpu_memory_info()['usage_percent']:.1%}")
    
    def load_model(self, model_name: str, model_instance: Any) -> bool:
        """Load a model with memory management"""
        print(f"🔄 Loading model: {model_name}")
        
        # Check if already loaded
        if model_name in self.model_registry:
            print(f"✅ Model {model_name} already loaded")
            return True
        
        # Check memory availability
        if not self.can_load_model(model_name):
            print(f"⚠️ Insufficient memory for {model_name}, performing cleanup...")
            self.emergency_cleanup()
            
            # Try again after cleanup
            if not self.can_load_model(model_name):
                print(f"❌ Still insufficient memory for {model_name}")
                return False
        
        try:
            # Load the model
            self.model_registry[model_name] = model_instance
            self.current_loaded_models.append(model_name)
            
            # Update memory usage
            memory_info = self.get_gpu_memory_info()
            self.model_memory_usage[model_name] = memory_info["allocated"]
            
            print(f"✅ Model {model_name} loaded successfully")
            print(f"   GPU memory usage: {memory_info['usage_percent']:.1%}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self.model_registry:
            print(f"🗑️ Unloading model: {model_name}")
            
            # Remove from registry
            del self.model_registry[model_name]
            if model_name in self.current_loaded_models:
                self.current_loaded_models.remove(model_name)
            if model_name in self.model_memory_usage:
                del self.model_memory_usage[model_name]
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            
            print(f"✅ Model {model_name} unloaded")
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model"""
        return self.model_registry.get(model_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded"""
        return model_name in self.model_registry
    
    @contextmanager
    def model_context(self, model_name: str, model_instance: Any):
        """Context manager for temporary model loading"""
        loaded = self.load_model(model_name, model_instance)
        try:
            yield loaded
        finally:
            if loaded:
                self.unload_model(model_name)
    
    def optimize_memory_for_models(self, required_models: List[str]):
        """Optimize memory usage for a set of required models"""
        print(f"🔧 Optimizing memory for models: {required_models}")
        
        # Calculate total memory needed
        total_required = sum(self.model_memory_estimates.get(m, 2.0) for m in required_models)
        memory_info = self.get_gpu_memory_info()
        
        print(f"   Required memory: {total_required:.1f}GB")
        print(f"   Available memory: {memory_info['free']:.1f}GB")
        
        # If we need more memory than available, unload unnecessary models
        if total_required > memory_info["free"]:
            models_to_unload = [m for m in self.current_loaded_models if m not in required_models]
            for model_name in models_to_unload:
                self.unload_model(model_name)
        
        # Emergency cleanup if still needed
        if self.get_gpu_memory_info()["usage_percent"] > self.emergency_threshold:
            self.emergency_cleanup()
    
    def get_memory_status(self) -> str:
        """Get a formatted memory status string"""
        info = self.get_gpu_memory_info()
        return (f"GPU Memory: {info['allocated']:.1f}GB/{info['total']:.1f}GB "
                f"({info['usage_percent']:.1%}) | Loaded: {len(self.current_loaded_models)} models")


# Global memory manager instance
memory_manager = GPUMemoryManager()


def get_memory_manager() -> GPUMemoryManager:
    """Get the global memory manager instance"""
    return memory_manager

