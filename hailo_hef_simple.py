"""
Simple Hailo HEF Model Inference Example
Loads a YOLOv8n HEF model and runs inference with timing
"""

import cv2
import numpy as np
import time
from pathlib import Path

try:
    from hailo_platform import (
        HEF, VDevice, ConfigureParams, InputVStreamParams, 
        OutputVStreamParams, HailoStreamInterface, InferVStreams
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Error: hailo_platform not installed")


class HailoInference:
    def __init__(self, hef_path, image_path=None):
        """
        Initialize Hailo inference engine
        
        Args:
            hef_path: Path to .hef model file
            image_path: Path to test image (optional, creates dummy if None)
        """
        self.hef_path = hef_path
        
        # Load or create test image
        if image_path and Path(image_path).exists():
            self.image = cv2.imread(image_path)
            print(f"✓ Loaded image: {image_path}")
        else:
            # Create dummy 640x640 image
            self.image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            print("✓ Using dummy 640x640 image")
        
        print(f"✓ Image shape: {self.image.shape}")
    
    def preprocess(self, image, input_shape):
        """
        Preprocess image for HEF model
        
        Args:
            image: OpenCV image (BGR)
            input_shape: Expected input shape (H, W, C)
            
        Returns:
            Preprocessed image as uint8
        """
        h, w, c = input_shape
        
        # Resize to model input size
        resized = cv2.resize(image, (w, h))
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        return rgb.astype(np.uint8)
    
    def run(self, num_warmup=5, num_iterations=50):
        """
        Run inference on HEF model
        
        Args:
            num_warmup: Number of warmup iterations
            num_iterations: Number of benchmark iterations
        """
        if not HAILO_AVAILABLE:
            print("Error: Hailo not available")
            return
        
        try:
            print("\n" + "="*60)
            print("HAILO HEF INFERENCE")
            print("="*60)
            
            # Load HEF
            print("\n[1/5] Loading HEF model...")
            load_start = time.perf_counter()
            hef = HEF(self.hef_path)
            load_time = (time.perf_counter() - load_start) * 1000
            print(f"✓ Model loaded in {load_time:.2f} ms")
            
            # Get model info
            print("\n[2/5] Getting model information...")
            input_infos = hef.get_input_vstream_infos()
            output_infos = hef.get_output_vstream_infos()
            
            input_info = input_infos[0]
            input_shape = input_info.shape
            input_name = input_info.name
            
            print(f"✓ Input shape: {input_shape}")
            print(f"✓ Input name: {input_name}")
            print(f"✓ Number of inputs: {len(input_infos)}")
            print(f"✓ Number of outputs: {len(output_infos)}")
            
            # Preprocess image
            print("\n[3/5] Preprocessing image...")
            input_data = self.preprocess(self.image, input_shape)
            print(f"✓ Preprocessed shape: {input_data.shape}, dtype: {input_data.dtype}")
            
            # Initialize device and run inference
            print("\n[4/5] Initializing Hailo device...")
            with VDevice() as device:
                # Configure network
                config_params = ConfigureParams.create_from_hef(hef)
                network_groups = device.configure(hef, config_params)
                network_group = network_groups[0]
                
                print(f"✓ Device configured")
                
                # Create vstream parameters
                input_params = InputVStreamParams.make(network_group, quantized=True)
                output_params = OutputVStreamParams.make(network_group, quantized=False)
                
                # Run inference
                print("\n[5/5] Running inference...")
                with InferVStreams(network_group, input_params, output_params) as infer:
                    
                    # Warmup
                    print(f"  Warming up ({num_warmup} iterations)...")
                    for i in range(num_warmup):
                        _ = infer.infer({input_name: input_data})
                    
                    # Benchmark
                    print(f"  Benchmarking ({num_iterations} iterations)...")
                    times = []
                    
                    for i in range(num_iterations):
                        t_start = time.perf_counter()
                        output = infer.infer({input_name: input_data})
                        t_end = time.perf_counter()
                        
                        times.append((t_end - t_start) * 1000)  # Convert to ms
                        
                        if (i + 1) % max(1, num_iterations // 4) == 0:
                            print(f"    Progress: {i + 1}/{num_iterations}")
            
            # Print results
            self.print_results(times, load_time, input_shape)
            return times
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def print_results(self, times, load_time, input_shape):
        """Print benchmark results"""
        import statistics
        
        avg = statistics.mean(times)
        median = statistics.median(times)
        min_time = min(times)
        max_time = max(times)
        std = statistics.stdev(times) if len(times) > 1 else 0
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        fps = 1000.0 / avg
        
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        print(f"Model Load Time:         {load_time:.2f} ms")
        print(f"\nInference Latency:")
        print(f"  Average:               {avg:.2f} ms")
        print(f"  Median:                {median:.2f} ms")
        print(f"  Min:                   {min_time:.2f} ms")
        print(f"  Max:                   {max_time:.2f} ms")
        print(f"  Std Dev:               {std:.2f} ms")
        print(f"  P95:                   {p95:.2f} ms")
        print(f"  P99:                   {p99:.2f} ms")
        print(f"\nThroughput:              {fps:.2f} FPS")
        print(f"Input Shape:             {input_shape}")
        print(f"Total Iterations:        {len(times)}")
        print("="*60)


def main():
    # Configuration
    HEF_MODEL_PATH = "yolov8n.hef"  # Path to your .hef file
    IMAGE_PATH = None  # Optional: path to test image
    
    # Create inference engine
    hailo = HailoInference(HEF_MODEL_PATH, IMAGE_PATH)
    
    # Run inference
    hailo.run(num_warmup=5, num_iterations=50)


if __name__ == "__main__":
    main()