import time
import numpy as np
import cv2
from pathlib import Path
import statistics
import json
from datetime import datetime

# For CPU evaluation
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not installed. Install with: pip install ultralytics")

# For Hailo NPU evaluation
try:
    from hailo_platform import (HEF, ConfigureParams, FormatType, HailoStreamInterface,
                                 InferVStreams, InputVStreamParams, OutputVStreamParams,
                                 HailoSchedulingAlgorithm, VDevice)
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Warning: hailo_platform not installed. NPU evaluation will be skipped.")

class YOLOEvaluator:
    def __init__(self, pt_model_path, hef_model_path, test_image_path=None, 
                 num_warmup=10, num_iterations=100):
        """
        Initialize YOLOv8n evaluator for CPU and Hailo NPU
        
        Args:
            pt_model_path: Path to .pt model file
            hef_model_path: Path to .hef model file
            test_image_path: Path to test image (optional)
            num_warmup: Number of warmup iterations
            num_iterations: Number of test iterations
        """
        self.pt_model_path = pt_model_path
        self.hef_model_path = hef_model_path
        self.test_image_path = test_image_path
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.results = {}
        
        # Prepare test image or dummy input
        if test_image_path and Path(test_image_path).exists():
            self.test_image = cv2.imread(test_image_path)
            print(f"Loaded test image: {test_image_path}")
        else:
            # Create dummy image (640x640 RGB)
            self.test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            print("Using dummy test image (640x640)")
    
    def evaluate_cpu(self):
        """Evaluate YOLOv8n on CPU using .pt model"""
        print("\n" + "="*70)
        print("EVALUATING YOLOV8N ON CPU (.pt model)")
        print("="*70)
        
        if not ULTRALYTICS_AVAILABLE:
            print("Error: ultralytics not installed")
            return None
        
        # Measure model loading time
        print("\n[1/4] Loading model...")
        load_start = time.perf_counter()
        model = YOLO(self.pt_model_path)
        load_end = time.perf_counter()
        load_time = (load_end - load_start) * 1000
        
        print(f"✓ Model loaded in {load_time:.2f} ms")
        
        # Warmup
        print(f"\n[2/4] Warming up ({self.num_warmup} iterations)...")
        for i in range(self.num_warmup):
            _ = model.predict(self.test_image, verbose=False, device='cpu')
            if (i + 1) % 5 == 0:
                print(f"  Warmup progress: {i + 1}/{self.num_warmup}")
        
        # Measure inference time
        print(f"\n[3/4] Running inference ({self.num_iterations} iterations)...")
        inference_times = []
        preprocessing_times = []
        postprocessing_times = []
        
        for i in range(self.num_iterations):
            # Measure total inference time
            start_time = time.perf_counter()
            results = model.predict(self.test_image, verbose=False, device='cpu')
            end_time = time.perf_counter()
            
            inference_time = (end_time - start_time) * 1000
            inference_times.append(inference_time)
            
            # Extract preprocessing and postprocessing times if available
            if hasattr(results[0], 'speed'):
                preprocessing_times.append(results[0].speed.get('preprocess', 0))
                postprocessing_times.append(results[0].speed.get('postprocess', 0))
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{self.num_iterations}")
        
        # Get detection results from last inference
        print("\n[4/4] Analyzing results...")
        final_results = model.predict(self.test_image, verbose=False, device='cpu')[0]
        num_detections = len(final_results.boxes)
        
        # Calculate statistics
        stats = {
            'model_type': 'YOLOv8n (PyTorch)',
            'device': 'CPU',
            'model_path': self.pt_model_path,
            'model_load_time_ms': load_time,
            'avg_inference_time_ms': statistics.mean(inference_times),
            'min_inference_time_ms': min(inference_times),
            'max_inference_time_ms': max(inference_times),
            'std_inference_time_ms': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
            'median_inference_time_ms': statistics.median(inference_times),
            'p95_inference_time_ms': np.percentile(inference_times, 95),
            'p99_inference_time_ms': np.percentile(inference_times, 99),
            'throughput_fps': 1000 / statistics.mean(inference_times),
            'num_detections': num_detections,
            'avg_preprocessing_time_ms': statistics.mean(preprocessing_times) if preprocessing_times else 0,
            'avg_postprocessing_time_ms': statistics.mean(postprocessing_times) if postprocessing_times else 0,
            'input_shape': self.test_image.shape,
            'num_iterations': self.num_iterations,
        }
        
        self.print_results(stats)
        self.results['cpu'] = stats
        
        return stats, final_results
    
    def evaluate_hailo_npu(self):
        """Evaluate YOLOv8n on Hailo NPU 8L using .hef model"""
        print("\n" + "="*70)
        print("EVALUATING YOLOV8N ON HAILO NPU 8L (.hef model)")
        print("="*70)
        
        if not HAILO_AVAILABLE:
            print("Error: hailo_platform not installed")
            print("Install with: pip install hailo-platform")
            return None
        
        try:
            # Measure model loading time
            print("\n[1/5] Loading HEF model...")
            load_start = time.perf_counter()
            
            # Initialize Hailo device
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            
            with VDevice(params) as target:
                hef = HEF(self.hef_model_path)
                load_end = time.perf_counter()
                load_time = (load_end - load_start) * 1000
                print(f"✓ HEF model loaded in {load_time:.2f} ms")
                
                # Configure network
                print("\n[2/5] Configuring network...")
                configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
                network_groups = target.configure(hef, configure_params)
                network_group = network_groups[0]
                network_group_params = network_group.create_params()
                
                # Get input/output specifications
                input_vstream_infos = hef.get_input_vstream_infos()
                output_vstream_infos = hef.get_output_vstream_infos()
                
                print(f"✓ Found {len(input_vstream_infos)} input stream(s)")
                print(f"✓ Found {len(output_vstream_infos)} output stream(s)")
                
                # Get first input stream info
                input_vstream_info = input_vstream_infos[0]
                
                # Get input shape and frame size
                input_shape = input_vstream_info.shape
                input_format = input_vstream_info.format.type
                
                # Calculate frame size: height * width * channels * bytes_per_element
                h, w, c = input_shape
                bytes_per_element = 1 if input_format == FormatType.UINT8 else 4
                frame_size = h * w * c * bytes_per_element
                
                print(f"✓ Input shape: {input_shape}")
                print(f"✓ Input format: {input_format}")
                print(f"✓ Frame size: {frame_size} bytes ({h}x{w}x{c}x{bytes_per_element})")
                print(f"✓ Number of output layers: {len(output_vstream_infos)}")
                
                # Determine quantization based on input format
                input_quantized = (input_format == FormatType.UINT8)
                input_format_type = FormatType.UINT8 if input_quantized else FormatType.FLOAT32
                
                # Output always uses FLOAT32 for YOLOv8
                output_quantized = False
                output_format_type = FormatType.FLOAT32
                
                print(f"✓ Input: quantized={input_quantized}, format_type={input_format_type}")
                print(f"✓ Output: quantized={output_quantized}, format_type={output_format_type}")
                
                # Create input/output vstream parameters
                input_vstreams_params = InputVStreamParams.make(network_group, quantized=input_quantized, format_type=input_format_type)
                output_vstreams_params = OutputVStreamParams.make(network_group, quantized=output_quantized, format_type=output_format_type)
                
                # Print output stream info for debugging
                print(f"\nOutput streams information:")
                for i, output_info in enumerate(output_vstream_infos):
                    print(f"  Output {i}: shape={output_info.shape}, format={output_info.format.type}")
                
                # Prepare output buffers
                print(f"\nPreparing output buffers...")
                output_buffers = self.prepare_output_buffers(output_vstream_infos)
                
                # Prepare input image with correct size
                print("\n[3/5] Preprocessing input image...")
                input_data = self.preprocess_for_hailo(self.test_image, input_vstream_info, frame_size)
                print(f"✓ Preprocessed input shape: {input_data.shape}, dtype: {input_data.dtype}")
                print(f"✓ Input data size: {input_data.nbytes} bytes (expected: {frame_size} bytes)")
                
                # Verify size matches
                if input_data.nbytes != frame_size:
                    print(f"⚠ Warning: Input data size mismatch. Adjusting...")
                    # Flatten and trim/pad to exact frame size
                    input_data = input_data.flatten()
                    if len(input_data) > frame_size:
                        input_data = input_data[:frame_size]
                    elif len(input_data) < frame_size:
                        input_data = np.pad(input_data, (0, frame_size - len(input_data)))
                    # Reshape back
                    input_data = input_data.astype(np.uint8 if input_quantized else np.float32)
                
                # Warmup
                print(f"\n[4/5] Warming up ({self.num_warmup} iterations)...")
                try:
                    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                        for i in range(self.num_warmup):
                            try:
                                _ = infer_pipeline.infer({input_vstream_info.name: input_data})
                                if (i + 1) % 5 == 0:
                                    print(f"  Warmup progress: {i + 1}/{self.num_warmup}")
                            except Exception as e:
                                print(f"  Warmup iteration {i+1} error: {e}")
                                if i == 0:  # Fail fast on first iteration
                                    raise
                        
                        # Measure inference time
                        print(f"\n[5/5] Running inference ({self.num_iterations} iterations)...")
                        inference_times = []
                        all_outputs = []
                        
                        for i in range(self.num_iterations):
                            start_time = time.perf_counter()
                            output = infer_pipeline.infer({input_vstream_info.name: input_data})
                            end_time = time.perf_counter()
                            
                            inference_time = (end_time - start_time) * 1000
                            inference_times.append(inference_time)
                            
                            if i == self.num_iterations - 1:
                                all_outputs = output
                            
                            if (i + 1) % 20 == 0:
                                print(f"  Progress: {i + 1}/{self.num_iterations}")
                except MemoryError as me:
                    print(f"Memory allocation error: {me}")
                    print("This may be due to insufficient GPU/system memory.")
                    print("Try reducing num_iterations or using a smaller batch size.")
                    raise
                
                # Calculate statistics
                stats = {
                    'model_type': 'YOLOv8n (HEF)',
                    'device': 'Hailo NPU 8L',
                    'model_path': self.hef_model_path,
                    'model_load_time_ms': load_time,
                    'avg_inference_time_ms': statistics.mean(inference_times),
                    'min_inference_time_ms': min(inference_times),
                    'max_inference_time_ms': max(inference_times),
                    'std_inference_time_ms': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                    'median_inference_time_ms': statistics.median(inference_times),
                    'p95_inference_time_ms': np.percentile(inference_times, 95),
                    'p99_inference_time_ms': np.percentile(inference_times, 99),
                    'throughput_fps': 1000 / statistics.mean(inference_times),
                    'input_shape': input_shape,
                    'frame_size': frame_size,
                    'num_output_layers': len(output_vstream_infos),
                    'num_iterations': self.num_iterations,
                }
                
                self.print_results(stats)
                self.results['hailo_npu'] = stats
                
                return stats, all_outputs
                
        except Exception as e:
            print(f"Error during Hailo NPU evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_output_buffers(self, output_vstream_infos):
        """
        Prepare output buffers with correct shapes and dtypes
        
        Args:
            output_vstream_infos: List of output vstream information objects
            
        Returns:
            Dictionary with output buffer names and arrays
        """
        output_buffers = {}
        for output_info in output_vstream_infos:
            shape = output_info.shape
            # Determine dtype based on format type
            if output_info.format.type == FormatType.UINT8:
                dtype = np.uint8
            else:
                dtype = np.float32
            
            # Create buffer with exact shape
            buffer = np.zeros(shape, dtype=dtype)
            output_buffers[output_info.name] = buffer
            print(f"  Created output buffer: {output_info.name}, shape={shape}, dtype={dtype}")
        
        return output_buffers
    
        """
        Preprocess image for Hailo NPU with correct frame size
        
        Args:
            image: Input image (BGR format from cv2)
            input_vstream_info: Input vstream information from HEF
            frame_size: Expected frame size in bytes
        """
        # Get input shape (height, width, channels)
        height, width, channels = input_vstream_info.shape
        
        # Resize image to match expected dimensions
        resized = cv2.resize(image, (width, height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Check if quantized input is expected
        if input_vstream_info.format.type == FormatType.UINT8:
            # Keep as uint8 [0-255]
            result = rgb_image.astype(np.uint8)
        else:
            # Normalize to [0, 1] for float32
            result = rgb_image.astype(np.float32) / 255.0
        
        # Verify the result matches frame size
        if result.nbytes == frame_size:
            return result.flatten()
        
        # If sizes don't match, try to reshape/adjust
        print(f"  Frame size mismatch: got {result.nbytes}, expected {frame_size}")
        
        # Create a buffer of the exact size needed
        dtype = np.uint8 if input_vstream_info.format.type == FormatType.UINT8 else np.float32
        buffer = np.zeros(frame_size // (1 if dtype == np.uint8 else 4), dtype=dtype)
        
        # Copy data into buffer
        flat_result = result.flatten()
        min_size = min(len(flat_result), len(buffer))
        buffer[:min_size] = flat_result[:min_size]
        
        return buffer
    
    def print_results(self, stats):
        """Print formatted results"""
        print("\n" + "-"*70)
        print(f"RESULTS: {stats['device']}")
        print("-"*70)
        print(f"Model Type:              {stats['model_type']}")
        print(f"Model Path:              {stats['model_path']}")
        print(f"Model Load Time:         {stats['model_load_time_ms']:.2f} ms")
        print(f"\nInference Performance:")
        print(f"  Average Time:          {stats['avg_inference_time_ms']:.2f} ms")
        print(f"  Median Time:           {stats['median_inference_time_ms']:.2f} ms")
        print(f"  Min Time:              {stats['min_inference_time_ms']:.2f} ms")
        print(f"  Max Time:              {stats['max_inference_time_ms']:.2f} ms")
        print(f"  Std Deviation:         {stats['std_inference_time_ms']:.2f} ms")
        print(f"  95th Percentile:       {stats['p95_inference_time_ms']:.2f} ms")
        print(f"  99th Percentile:       {stats['p99_inference_time_ms']:.2f} ms")
        print(f"  Throughput:            {stats['throughput_fps']:.2f} FPS")
        
        if 'num_detections' in stats:
            print(f"\nDetection Results:")
            print(f"  Number of Detections:  {stats['num_detections']}")
        
        if 'frame_size' in stats:
            print(f"\nBuffer Information:")
            print(f"  Frame Size:            {stats['frame_size']} bytes")
        
        print(f"\nTest Configuration:")
        print(f"  Input Shape:           {stats['input_shape']}")
        print(f"  Iterations:            {stats['num_iterations']}")
        print("-"*70)
    
    def compare_results(self):
        """Generate comparison between CPU and NPU"""
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON: CPU vs HAILO NPU 8L")
        print("="*70)
        
        if 'cpu' not in self.results or 'hailo_npu' not in self.results:
            print("Both CPU and NPU evaluations needed for comparison")
            return
        
        cpu = self.results['cpu']
        npu = self.results['hailo_npu']
        
        # Calculate speedup
        inference_speedup = cpu['avg_inference_time_ms'] / npu['avg_inference_time_ms']
        load_time_ratio = cpu['model_load_time_ms'] / npu['model_load_time_ms']
        fps_improvement = (npu['throughput_fps'] - cpu['throughput_fps']) / cpu['throughput_fps'] * 100
        
        print(f"\n{'Metric':<30} {'CPU':<20} {'Hailo NPU 8L':<20} {'Improvement':<15}")
        print("-"*85)
        print(f"{'Model Load Time':<30} {cpu['model_load_time_ms']:>8.2f} ms      {npu['model_load_time_ms']:>8.2f} ms      {load_time_ratio:>8.2f}x")
        print(f"{'Avg Inference Time':<30} {cpu['avg_inference_time_ms']:>8.2f} ms      {npu['avg_inference_time_ms']:>8.2f} ms      {inference_speedup:>8.2f}x")
        print(f"{'Min Inference Time':<30} {cpu['min_inference_time_ms']:>8.2f} ms      {npu['min_inference_time_ms']:>8.2f} ms")
        print(f"{'Max Inference Time':<30} {cpu['max_inference_time_ms']:>8.2f} ms      {npu['max_inference_time_ms']:>8.2f} ms")
        print(f"{'Throughput (FPS)':<30} {cpu['throughput_fps']:>8.2f}         {npu['throughput_fps']:>8.2f}         {fps_improvement:>7.1f}%")
        print(f"{'Latency P95':<30} {cpu['p95_inference_time_ms']:>8.2f} ms      {npu['p95_inference_time_ms']:>8.2f} ms")
        print(f"{'Latency P99':<30} {cpu['p99_inference_time_ms']:>8.2f} ms      {npu['p99_inference_time_ms']:>8.2f} ms")
        
        print("\n" + "="*70)
        print(f"SUMMARY:")
        print(f"  NPU is {inference_speedup:.2f}x faster than CPU for inference")
        print(f"  NPU achieves {fps_improvement:+.1f}% higher throughput")
        print(f"  NPU model loads {load_time_ratio:.2f}x {'faster' if load_time_ratio > 1 else 'slower'} than CPU model")
        print("="*70)
    
    def save_results(self, output_file='evaluation_results.json'):
        """Save results to JSON file"""
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'test_configuration': {
                'pt_model': self.pt_model_path,
                'hef_model': self.hef_model_path,
                'test_image': self.test_image_path,
                'num_iterations': self.num_iterations,
                'num_warmup': self.num_warmup,
            },
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to {output_file}")

def main():
    # Configuration
    PT_MODEL_PATH = "yolov8n.pt"  # Path to your .pt model
    HEF_MODEL_PATH = "yolov8n.hef"  # Path to your .hef model
    TEST_IMAGE_PATH = None  # Optional: path to test image, None for dummy image
    
    # Initialize evaluator
    evaluator = YOLOEvaluator(
        pt_model_path=PT_MODEL_PATH,
        hef_model_path=HEF_MODEL_PATH,
        test_image_path=TEST_IMAGE_PATH,
        num_warmup=10,
        num_iterations=100
    )
    
    print("="*70)
    print("YOLOv8n EVALUATION: CPU vs HAILO NPU 8L")
    print("="*70)
    print(f"PT Model:  {PT_MODEL_PATH}")
    print(f"HEF Model: {HEF_MODEL_PATH}")
    print(f"Warmup:    {evaluator.num_warmup} iterations")
    print(f"Test:      {evaluator.num_iterations} iterations")
    print("="*70)
    
    # Evaluate on CPU
    cpu_stats, cpu_results = evaluator.evaluate_cpu()
    
    # Evaluate on Hailo NPU
    npu_stats = evaluator.evaluate_hailo_npu()
    
    # Compare results
    if cpu_stats and npu_stats:
        evaluator.compare_results()
        evaluator.save_results('yolov8n_evaluation_results.json')
    
    print("\n✓ Evaluation complete!")

if __name__ == "__main__":
    main()