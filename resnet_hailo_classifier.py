#!/usr/bin/env python3
"""
ResNet Image Classification on Hailo 8L NPU
Classifies images using ResNet .hef model without requiring Hailo Model Zoo
"""

import numpy as np
from PIL import Image
from hailo_platform import (HEF, Device, VDevice, HailoStreamInterface, 
                            InferVStreams, ConfigureParams)
import requests
import json

# ImageNet class labels (1000 classes)
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

class ResNetClassifier:
    def __init__(self, hef_path):
        """Initialize ResNet classifier with .hef model"""
        self.hef_path = hef_path
        self.hef = None
        self.target = None
        self.network_group = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.class_labels = self.load_imagenet_labels()
        
    def load_imagenet_labels(self):
        """Download and load ImageNet class labels"""
        try:
            print("Loading ImageNet class labels...")
            response = requests.get(IMAGENET_LABELS_URL, timeout=10)
            labels = response.json()
            print(f"Loaded {len(labels)} class labels")
            return labels
        except Exception as e:
            print(f"Warning: Could not load labels: {e}")
            print("Using class indices instead")
            return [f"Class_{i}" for i in range(1000)]
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess image for ResNet model
        ResNet expects 224x224 RGB images with ImageNet normalization
        """
        print(f"\nPreprocessing image: {image_path}")
        
        # Load and resize image
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        
        # Convert to numpy array [0, 255]
        img_array = np.array(img, dtype=np.float32)
        
        # ImageNet normalization
        # Values from PyTorch/TensorFlow ImageNet preprocessing
        img_array = img_array / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Convert to NHWC format if needed (batch, height, width, channels)
        # Some models may need HWC format without batch dimension
        print(f"Preprocessed image shape: {img_array.shape}")
        
        return img_array
    
    def setup_model(self):
        """Setup Hailo device and load model"""
        print(f"\nLoading HEF model: {self.hef_path}")
        self.hef = HEF(self.hef_path)
        
        # Scan for devices
        devices = Device.scan()
        print(f"Found {len(devices)} Hailo device(s)")
        
        if len(devices) == 0:
            raise RuntimeError("No Hailo devices found! Please check your Hailo 8L connection.")
        
        # Create VDevice
        print("Initializing Hailo 8L device...")
        self.target = VDevice(device_ids=devices)
        
        # Configure network
        print("Configuring network...")
        configure_params = ConfigureParams.create_from_hef(
            self.hef, 
            interface=HailoStreamInterface.PCIe
        )
        network_groups = self.target.configure(self.hef, configure_params)
        self.network_group = network_groups[0]
        
        # Get network parameters
        network_group_params = self.network_group.create_params()
        self.input_vstreams_params = network_group_params.input_vstreams_params
        self.output_vstreams_params = network_group_params.output_vstreams_params
        
        # Print model information
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Network: {self.network_group.name}")
        print(f"\nInput Layers: {len(self.input_vstreams_params)}")
        for i, param in enumerate(self.input_vstreams_params):
            print(f"  [{i}] Name: {param.name}")
            print(f"      Shape: {param.shape}")
            print(f"      Format: {param.format_type}")
        
        print(f"\nOutput Layers: {len(self.output_vstreams_params)}")
        for i, param in enumerate(self.output_vstreams_params):
            print(f"  [{i}] Name: {param.name}")
            print(f"      Shape: {param.shape}")
            print(f"      Format: {param.format_type}")
        print("="*60)
    
    def classify(self, image_path):
        """Run classification on an image"""
        
        # Get input shape from model
        input_shape = self.input_vstreams_params[0].shape
        print(f"\nModel expects input shape: {input_shape}")
        
        # Determine target size (usually height, width from shape)
        if len(input_shape) == 4:  # NHWC format
            target_size = (input_shape[1], input_shape[2])
        elif len(input_shape) == 3:  # HWC format
            target_size = (input_shape[0], input_shape[1])
        else:
            target_size = (224, 224)  # Default for ResNet
        
        # Preprocess image
        input_data = self.preprocess_image(image_path, target_size)
        
        # Ensure correct shape for model
        if len(input_shape) == 3 and len(input_data.shape) == 3:
            # Model expects HWC, data is HWC - good
            pass
        elif len(input_shape) == 4 and len(input_data.shape) == 3:
            # Model expects NHWC, add batch dimension
            input_data = np.expand_dims(input_data, axis=0)
        
        print(f"Final input shape: {input_data.shape}")
        
        # Prepare input dictionary
        input_name = self.input_vstreams_params[0].name
        input_dict = {input_name: input_data}
        
        # Run inference
        print("\n" + "="*60)
        print("RUNNING INFERENCE ON HAILO 8L NPU...")
        print("="*60)
        
        import time
        start_time = time.time()
        
        with InferVStreams(self.network_group, 
                          self.input_vstreams_params, 
                          self.output_vstreams_params) as infer_pipeline:
            output_dict = infer_pipeline.infer(input_dict)
        
        inference_time = (time.time() - start_time) * 1000
        print(f"✓ Inference completed in {inference_time:.2f} ms")
        
        # Process output
        output_name = list(output_dict.keys())[0]
        output_data = output_dict[output_name]
        
        print(f"\nOutput shape: {output_data.shape}")
        print(f"Output dtype: {output_data.dtype}")
        
        # Flatten output to 1D array
        predictions = output_data.flatten()
        
        # Apply softmax if needed (to get probabilities)
        predictions = self.softmax(predictions)
        
        return predictions
    
    def softmax(self, x):
        """Apply softmax to convert logits to probabilities"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def print_predictions(self, predictions, top_k=5):
        """Print top K predictions"""
        print("\n" + "="*60)
        print(f"TOP {top_k} PREDICTIONS")
        print("="*60)
        
        # Get top K indices
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        for rank, idx in enumerate(top_indices, 1):
            confidence = predictions[idx] * 100
            class_name = self.class_labels[idx] if idx < len(self.class_labels) else f"Class_{idx}"
            print(f"{rank}. {class_name}")
            print(f"   Confidence: {confidence:.2f}%")
            print(f"   Class ID: {idx}")
            if rank < top_k:
                print()
    
    def cleanup(self):
        """Release resources"""
        if self.target:
            self.target.release()
            print("\n✓ Released Hailo device")
    
    def run(self, image_path, top_k=5):
        """Complete classification workflow"""
        try:
            # Setup model
            self.setup_model()
            
            # Run classification
            predictions = self.classify(image_path)
            
            # Display results
            self.print_predictions(predictions, top_k)
            
            return predictions
            
        except Exception as e:
            print(f"\n✗ Error during classification: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            self.cleanup()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Classify images using ResNet on Hailo 8L NPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resnet_classifier.py --hef resnet_v1_50.hef --image dog.jpg
  python resnet_classifier.py --hef resnet_v1_50.hef --image cat.jpg --top 10
        """
    )
    
    parser.add_argument("--hef", required=True, 
                       help="Path to ResNet .hef model file")
    parser.add_argument("--image", required=True, 
                       help="Path to input image (jpg, png, etc.)")
    parser.add_argument("--top", type=int, default=5, 
                       help="Number of top predictions to show (default: 5)")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("RESNET IMAGE CLASSIFICATION - HAILO 8L NPU")
    print("="*60)
    print(f"Model: {args.hef}")
    print(f"Image: {args.image}")
    print("="*60)
    
    # Create classifier and run
    classifier = ResNetClassifier(args.hef)
    predictions = classifier.run(args.image, top_k=args.top)
    
    if predictions is not None:
        print("\n✓ Classification completed successfully!")
    else:
        print("\n✗ Classification failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
