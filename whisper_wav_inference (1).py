#!/usr/bin/env python3
"""
Whisper Speech Recognition with WAV File Input for Hailo NPU
Processes entire audio file by splitting into 30-second chunks
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
    
    return audio

def split_audio_into_chunks(audio, chunk_size=N_SAMPLES):
    """
    Split audio into 30-second chunks for processing
    
    Args:
        audio: Full audio array
        chunk_size: Number of samples per chunk (default: 30 seconds)
    
    Returns:
        List of audio chunks
    """
    chunks = []
    num_chunks = int(np.ceil(len(audio) / chunk_size))
    
    print(f"\nSplitting audio into {num_chunks} chunks of 30 seconds each...")
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(audio))
        
        chunk = audio[start:end]
        
        # Pad last chunk if necessary
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
            print(f"Chunk {i+1}/{num_chunks}: Padded to 30 seconds")
        else:
            print(f"Chunk {i+1}/{num_chunks}: {len(chunk)/SAMPLE_RATE:.2f} seconds")
        
        chunks.append(chunk)
    
    return chunks

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

def process_audio_chunk(encoder_infer, decoder_infer, mel_features, chunk_idx):
    """
    Process a single audio chunk through encoder and decoder
    
    Args:
        encoder_infer: Encoder inference pipeline
        decoder_infer: Decoder inference pipeline
        mel_features: Mel spectrogram features for this chunk
        chunk_idx: Index of the current chunk
    
    Returns:
        Decoder output for this chunk
    """
    print(f"\n{'='*50}")
    print(f"Processing chunk {chunk_idx}...")
    print(f"{'='*50}")
    
    # Get input/output info
    input_vstream_info = encoder_infer.input_vstreams
    output_vstream_info = encoder_infer.output_vstreams
    
    # Prepare encoder input
    encoder_input = mel_features.astype(np.float32)
    
    # Create input dictionary
    input_data = {list(input_vstream_info.keys())[0]: encoder_input}
    
    # Run encoder inference
    print(f"Running encoder on chunk {chunk_idx}...")
    encoder_output = encoder_infer.infer(input_data)
    print(f"✓ Encoder completed for chunk {chunk_idx}")
    
    # Get decoder input/output info
    decoder_input_info = decoder_infer.input_vstreams
    decoder_output_info = decoder_infer.output_vstreams
    
    # Prepare decoder input
    # Note: This is simplified. Real Whisper decoding is autoregressive
    # Map encoder output to decoder input based on your model structure
    decoder_input = {}
    
    # Example: If your decoder expects the encoder output directly
    # Adjust this based on your actual model input names
    encoder_output_key = list(encoder_output.keys())[0]
    decoder_input_key = list(decoder_input_info.keys())[0]
    decoder_input[decoder_input_key] = encoder_output[encoder_output_key]
    
    # Run decoder inference
    print(f"Running decoder on chunk {chunk_idx}...")
    decoder_output = decoder_infer.infer(decoder_input)
    print(f"✓ Decoder completed for chunk {chunk_idx}")
    
    return decoder_output

def run_inference(encoder_hef_path, decoder_hef_path, audio_path):
    """
    Run Whisper inference on entire audio file
    
    Args:
        encoder_hef_path: Path to encoder HEF file
        decoder_hef_path: Path to decoder HEF file
        audio_path: Path to input WAV file
    """
    
    # Load audio and split into chunks
    audio = load_audio(audio_path)
    audio_chunks = split_audio_into_chunks(audio)
    
    print(f"\nTotal chunks to process: {len(audio_chunks)}")
    
    # Load HEF files
    print("\nLoading Hailo models...")
    encoder_hef = HEF(encoder_hef_path)
    decoder_hef = HEF(decoder_hef_path)
    
    # Store all results
    all_results = []
    
    # Create VDevice and run inference
    with VDevice() as target:
        print("Configuring encoder...")
        
        # Configure encoder
        encoder_params = ConfigureParams.create_from_hef(
            encoder_hef, 
            interface=HailoStreamInterface.PCIe
        )
        encoder_network_group = target.configure(encoder_hef, encoder_params)[0]
        
        # Configure decoder
        print("Configuring decoder...")
        decoder_params = ConfigureParams.create_from_hef(
            decoder_hef,
            interface=HailoStreamInterface.PCIe
        )
        decoder_network_group = target.configure(decoder_hef, decoder_params)[0]
        
        # Process all chunks
        with InferVStreams(encoder_network_group) as encoder_infer, \
             InferVStreams(decoder_network_group) as decoder_infer:
            
            # Print model info once
            input_vstream_info = encoder_infer.input_vstreams
            output_vstream_info = encoder_infer.output_vstreams
            
            print(f"\nEncoder input shape: {list(input_vstream_info.values())[0].shape}")
            print(f"Encoder output shape: {list(output_vstream_info.values())[0].shape}")
            
            decoder_input_info = decoder_infer.input_vstreams
            decoder_output_info = decoder_infer.output_vstreams
            
            print(f"Decoder input names: {list(decoder_input_info.keys())}")
            print(f"Decoder output names: {list(decoder_output_info.keys())}")
            
            # Process each chunk
            for idx, chunk in enumerate(audio_chunks, 1):
                # Convert chunk to mel spectrogram
                mel_features = audio_to_mel_spectrogram(chunk)
                
                # Process chunk
                chunk_output = process_audio_chunk(
                    encoder_infer, 
                    decoder_infer, 
                    mel_features, 
                    idx
                )
                
                all_results.append({
                    'chunk_idx': idx,
                    'output': chunk_output,
                    'duration': len(chunk) / SAMPLE_RATE
                })
            
            print(f"\n{'='*60}")
            print(f"✓ All {len(audio_chunks)} chunks processed successfully!")
            print(f"{'='*60}")
            
            return all_results

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
        results = run_inference(args.encoder, args.decoder, args.audio)
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total chunks processed: {len(results)}")
        print(f"Total audio duration: {sum(r['duration'] for r in results):.2f} seconds")
        print("="*60)
        print("\n✓ Inference completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
