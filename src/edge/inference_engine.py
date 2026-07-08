"""
Inference Engine Abstraction Layer
Supports multiple backends: ONNXRuntime, Hailo (future)

This abstraction allows seamless switching between inference backends
without modifying the detection pipeline.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from pathlib import Path

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNXRuntime not available")

# Hailo runtime will be imported when available
HAILO_AVAILABLE = False
try:
    # from hailo_platform import HailoRT
    # HAILO_AVAILABLE = True
    pass
except ImportError:
    pass


class InferenceEngine(ABC):
    """Abstract base class for inference engines"""
    
    @abstractmethod
    def load_model(self, model_path: Path):
        """Load model into memory"""
        pass
    
    @abstractmethod
    def predict(self, preprocessed_frame: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed frame"""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Return expected input shape (batch, channels, height, width)"""
        pass
    
    @abstractmethod
    def warmup(self):
        """Run warmup inference for performance optimization"""
        pass


class ONNXInferenceEngine(InferenceEngine):
    """ONNX Runtime inference engine (CPU optimized for Pi)"""
    
    def __init__(self, num_threads: int = 4):
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNXRuntime not installed. Install: pip install onnxruntime")
        
        self.session = None
        self.input_name = None
        self.output_names = None
        self.num_threads = num_threads
        
    def load_model(self, model_path: Path):
        """Load ONNX model with CPU-optimized session options"""
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Session options optimized for Raspberry Pi
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = self.num_threads
        sess_options.inter_op_num_threads = self.num_threads
        
        # Use CPUExecutionProvider only (no CUDA on Pi)
        providers = ['CPUExecutionProvider']
        
        # Load model
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✓ ONNX model loaded: {model_path.name}")
        print(f"  Input: {self.input_name}, Shape: {self.session.get_inputs()[0].shape}")
        print(f"  Providers: {self.session.get_providers()}")
        
    def predict(self, preprocessed_frame: np.ndarray) -> np.ndarray:
        """Run ONNX inference"""
        
        if self.session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed_frame})
        
        return outputs[0]  # Return first output (detection results)
    
    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape"""
        return tuple(self.session.get_inputs()[0].shape)
    
    def warmup(self, num_runs: int = 5):
        """Warmup inference engine"""
        
        batch, channels, height, width = self.get_input_shape()
        dummy_input = np.random.randn(batch, channels, height, width).astype(np.float32)
        
        print(f"Warming up ONNX engine ({num_runs} runs)...")
        for _ in range(num_runs):
            self.predict(dummy_input)
        print("✓ Warmup complete")


class HailoInferenceEngine(InferenceEngine):
    """
    Hailo-8 inference engine (placeholder for future implementation)
    
    To enable:
    1. Export ONNX model with static shape (640x640)
    2. Compile ONNX to HEF using Hailo Dataflow Compiler:
       hailomz compile yolov8n --ckpt best_ball_nano.onnx --hw-arch hailo8 --output best_ball_nano.hef
    3. Load HEF model with this engine
    """
    
    def __init__(self, device_id: int = 0):
        if not HAILO_AVAILABLE:
            raise RuntimeError(
                "Hailo runtime not available. "
                "Install Hailo SDK: https://hailo.ai/developer-zone/"
            )
        
        self.device_id = device_id
        self.network = None
        self.input_vstream = None
        self.output_vstream = None
        
    def load_model(self, model_path: Path):
        """Load Hailo HEF model"""
        
        # Placeholder implementation
        # Actual implementation will use Hailo SDK:
        # 
        # from hailo_platform import HailoRT, HEF, VDevice, InputVStreamParams, OutputVStreamParams
        # 
        # hef = HEF(str(model_path))
        # target = VDevice()
        # network_group = target.configure(hef)
        # self.network = network_group.get_network_group()
        
        raise NotImplementedError(
            "Hailo inference engine not yet implemented. "
            "Currently using ONNX runtime. "
            "Hailo integration will be added after HEF compilation."
        )
    
    def predict(self, preprocessed_frame: np.ndarray) -> np.ndarray:
        """Run Hailo inference"""
        raise NotImplementedError("Hailo engine not implemented yet")
    
    def get_input_shape(self) -> Tuple[int, int, int, int]:
        """Get expected input shape"""
        return (1, 3, 640, 640)  # Static shape for Hailo
    
    def warmup(self):
        """Warmup Hailo accelerator"""
        raise NotImplementedError("Hailo engine not implemented yet")


def create_inference_engine(engine_type: str, **kwargs) -> InferenceEngine:
    """
    Factory function to create inference engine
    
    Args:
        engine_type: 'onnx' or 'hailo'
        **kwargs: Engine-specific parameters
        
    Returns:
        Configured inference engine
    """
    
    if engine_type.lower() == 'onnx':
        return ONNXInferenceEngine(num_threads=kwargs.get('num_threads', 4))
    
    elif engine_type.lower() == 'hailo':
        return HailoInferenceEngine(device_id=kwargs.get('device_id', 0))
    
    else:
        raise ValueError(f"Unknown engine type: {engine_type}. Use 'onnx' or 'hailo'")
