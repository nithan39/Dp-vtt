#!/usr/bin/env python3
"""
Whisper Speech Recognition with WAV File Input for Hailo NPU
Modified to accept WAV files instead of live microphone input
"""

import argparse
import numpy as np
import librosa
from hailo_platform import (HEF, VDevice, HailoStreamInterface, 
                            InferVStreams, ConfigureParams, FormatType)

# Whisper model constants
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
CHUNK_LENGTH = 30  # seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples

def load_audio(audio_path):
    """
    Load and preprocess audio file for Whisper model
    
    Args:
        audio_path: Path to WAV file
    
    Returns:
        Audio array resampled to 16kHz
    """
    print(f"Loading audio from: {audio_path}")
    
    # Load audio file and resample to 16kHz
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    
    print(f"Audio loaded: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.2f} seconds")
    
    # Pad or trim to 30 seconds
    if len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
        print(f"Audio trimmed to 30 seconds")
    elif len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), mode='constant')
        print(f"Audio padded to 30 seconds")
    
    return audio

def audio_to_mel_spectrogram(audio):
    """
    Convert audio to mel spectrogram (Whisper preprocessing)
    
    Args:
        audio: Audio samples array
    
    Returns:
        Mel spectrogram features
    """
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=8000
    )
    
    # Convert to log scale (dB)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize
    log_mel = (log_mel + 80) / 80
    
    return log_mel.T  # Transpose to (time, mels)

def run_inference(encoder_hef_path, decoder_hef_path, audio_path):
    """
    Run Whisper inference on audio file
    
    Args:
        encoder_hef_path: Path to encoder HEF file
        decoder_hef_path: Path to decoder HEF file
        audio_path: Path to input WAV file
    """
    
    # Load audio and preprocess
    audio = load_audio(audio_path)
    mel_features = audio_to_mel_spectrogram(audio)
    
    print(f"Mel spectrogram shape: {mel_features.shape}")
    
    # Load HEF files
    print("\nLoading Hailo models...")
    encoder_hef = HEF(encoder_hef_path)
    decoder_hef = HEF(decoder_hef_path)
    
    # Create VDevice and run inference
    with VDevice() as target:
        print("Configuring encoder...")
        
        # Configure encoder
        encoder_params = ConfigureParams.create_from_hef(
            encoder_hef, 
            interface=HailoStreamInterface.PCIe
        )
        encoder_network_group = target.configure(encoder_hef, encoder_params)[0]
        
        # Run encoder inference
        with InferVStreams(encoder_network_group) as encoder_infer:
            print("\nRunning encoder inference...")
            
            # Get input/output info
            input_vstream_info = encoder_infer.input_vstreams
            output_vstream_info = encoder_infer.output_vstreams
            
            print(f"Encoder input shape: {list(input_vstream_info.values())[0].shape}")
            print(f"Encoder output shape: {list(output_vstream_info.values())[0].shape}")
            
            # Prepare input (may need reshaping based on your model)
            encoder_input = mel_features.astype(np.float32)
            
            # Create input dictionary
            input_data = {list(input_vstream_info.keys())[0]: encoder_input}
            
            # Run inference
            encoder_output = encoder_infer.infer(input_data)
            
            print("Encoder inference completed!")
            print(f"Encoder output keys: {encoder_output.keys()}")
        
        # Configure decoder
        print("\nConfiguring decoder...")
        decoder_params = ConfigureParams.create_from_hef(
            decoder_hef,
            interface=HailoStreamInterface.PCIe
        )
        decoder_network_group = target.configure(decoder_hef, decoder_params)[0]
        
        # Run decoder inference (this is simplified - actual Whisper decoding is iterative)
        with InferVStreams(decoder_network_group) as decoder_infer:
            print("Running decoder inference...")
            
            # Get decoder input/output info
            decoder_input_info = decoder_infer.input_vstreams
            decoder_output_info = decoder_infer.output_vstreams
            
            print(f"Decoder input names: {list(decoder_input_info.keys())}")
            print(f"Decoder output names: {list(decoder_output_info.keys())}")
            
            # Prepare decoder input
            # Note: This is simplified. Real Whisper decoding is autoregressive
            # You'll need to implement the full decoding loop
            
            decoder_input = {
                # Map encoder output to decoder input
                # This depends on your specific model structure
            }
            
            decoder_output = decoder_infer.infer(decoder_input)
            
            print("Decoder inference completed!")
            print(f"Output shape: {[v.shape for v in decoder_output.values()]}")
            
            return decoder_output

def main():
    parser = argparse.ArgumentParser(
        description='Run Whisper speech recognition on WAV file using Hailo NPU'
    )
    parser.add_argument(
        '--encoder', 
        type=str, 
        required=True,
        help='Path to Whisper encoder HEF file'
    )
    parser.add_argument(
        '--decoder',
        type=str,
        required=True,
        help='Path to Whisper decoder HEF file'
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to input WAV audio file'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Whisper Speech Recognition - WAV File Mode")
    print("="*60)
    print(f"Encoder HEF: {args.encoder}")
    print(f"Decoder HEF: {args.decoder}")
    print(f"Audio file: {args.audio}")
    print("="*60)
    
    try:
        output = run_inference(args.encoder, args.decoder, args.audio)
        print("\n✓ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
