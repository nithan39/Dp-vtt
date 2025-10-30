#!/usr/bin/env python3
"""
ResNet Inference Script for Hailo
Runs classification on an input image using a pre-compiled HEF model
"""

import numpy as np
import cv2
from hailo_platform import (HEF, ConfigureParams, FormatType, HailoSchedulingAlgorithm, 
                             HailoStreamInterface, InferVStreams, InputVStreamParams, 
                             OutputVStreamParams, VDevice)
import argparse
import sys

# ImageNet class labels (1000 classes)
# You can download the full list or use a subset
IMAGENET_CLASSES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
    "electric ray", "stingray", "cock", "hen", "ostrich"
    # ... (truncated for brevity - add all 1000 classes or load from file)
]

def load_imagenet_labels(labels_file=None):
    """Load ImageNet class labels"""
    if labels_file:
        with open(labels_file, 'r') as f:
            return [line.strip() for line in f.readlines()]
    return IMAGENET_CLASSES

def preprocess_image(image_path, input_height=224, input_width=224):
    """
    Preprocess image for ResNet:
    - Resize to model input size
    - Normalize (typical ImageNet normalization)
    - Convert to format expected by model
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, (input_width, input_height))
    
    # Normalize (ImageNet mean and std)
    img = img.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53])
    std = np.array([58.395, 57.12, 57.375])
    img = (img - mean) / std
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img

def postprocess_output(output, labels, top_k=5):
    """
    Process model output to get top-k predictions
    """
    # Flatten output if needed
    output = output.flatten()
    
    # Get top-k indices
    top_indices = np.argsort(output)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        class_name = labels[idx] if idx < len(labels) else f"Class_{idx}"
        confidence = output[idx]
        results.append((class_name, confidence, idx))
    
    return results

def run_inference(hef_path, image_path, labels_file=None):
    """
    Run inference using Hailo device
    """
    print(f"Loading HEF model: {hef_path}")
    
    # Load labels
    labels = load_imagenet_labels(labels_file)
    
    # Create VDevice (virtual device)
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    
    with VDevice(params) as target:
        # Load HEF
        hef = HEF(hef_path)
        
        # Configure network group
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()
        
        # Get input/output specs
        input_vstreams_params = InputVStreamParams.make(network_group, quantized=False, 
                                                         format_type=FormatType.FLOAT32)
        output_vstreams_params = OutputVStreamParams.make(network_group, quantized=False,
                                                           format_type=FormatType.FLOAT32)
        
        # Get input shape
        input_shape = input_vstreams_params[0].shape
        print(f"Model input shape: {input_shape}")
        
        # Preprocess image
        print(f"Processing image: {image_path}")
        input_data = preprocess_image(image_path, input_shape[0], input_shape[1])
        
        # Create input dictionary
        input_dict = {input_vstreams_params[0].name: input_data}
        
        # Run inference
        print("Running inference...")
        with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            with network_group.activate(network_group_params):
                output = infer_pipeline.infer(input_dict)
        
        # Get output
        output_name = list(output.keys())[0]
        output_data = output[output_name]
        
        # Postprocess
        results = postprocess_output(output_data, labels, top_k=5)
        
        # Print results
        print("\n" + "="*60)
        print("CLASSIFICATION RESULTS")
        print("="*60)
        for i, (class_name, confidence, idx) in enumerate(results, 1):
            print(f"{i}. {class_name:30s} - {confidence:.4f} (class {idx})")
        print("="*60)
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Run ResNet inference with Hailo')
    parser.add_argument('--hef', required=True, help='Path to HEF model file')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--labels', help='Path to ImageNet labels file (optional)')
    
    args = parser.parse_args()
    
    try:
        run_inference(args.hef, args.image, args.labels)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
