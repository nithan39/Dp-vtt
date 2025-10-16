import numpy as np
import librosa
from pathlib import Path
import argparse
from hailo_sdk_client import ClientRunner
import time

def load_and_preprocess_audio(audio_path, sr=16000):
    """Load and preprocess audio file to model input format."""
    # Load audio file
    waveform, sample_rate = librosa.load(audio_path, sr=sr)
    
    # Normalize audio
    waveform = waveform / np.max(np.abs(waveform))
    
    return waveform, sample_rate

def prepare_encoder_input(waveform, sr=16000):
    """Prepare audio input for Whisper encoder (mel-spectrogram)."""
    # Resample if needed
    if sr != 16000:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        fmin=0,
        fmax=8000
    )
    
    # Convert to log scale
    log_mel = np.log10(np.maximum(mel_spec, 1e-9))
    
    # Normalize
    log_mel = (log_mel + 4) / 4
    
    # Pad to standard length (3000 frames for Whisper)
    max_frames = 3000
    if log_mel.shape[1] < max_frames:
        log_mel = np.pad(log_mel, ((0, 0), (0, max_frames - log_mel.shape[1])), mode='constant')
    else:
        log_mel = log_mel[:, :max_frames]
    
    return np.expand_dims(log_mel, axis=0).astype(np.float32)

def run_encoder(encoder_runner, encoder_input):
    """Run inference on encoder model."""
    start_time = time.time()
    encoder_output = encoder_runner.predict(encoder_input)
    encoder_time = time.time() - start_time
    
    print(f"Encoder inference time: {encoder_time:.4f}s")
    print(f"Encoder output shape: {encoder_output[0].shape}")
    
    return encoder_output[0], encoder_time

def run_decoder(decoder_runner, encoder_output, max_tokens=448):
    """Run inference on decoder model with encoder output."""
    # Initialize decoder inputs (tokens, key-value cache, etc.)
    # This is a simplified version - adapt based on your specific decoder requirements
    decoder_input = np.zeros((1, 1), dtype=np.int64)  # Start token
    
    start_time = time.time()
    decoder_output = decoder_runner.predict([encoder_output, decoder_input])
    decoder_time = time.time() - start_time
    
    print(f"Decoder inference time: {decoder_time:.4f}s")
    print(f"Decoder output shape: {decoder_output[0].shape}")
    
    return decoder_output, decoder_time

def evaluate_model(encoder_hef_path, decoder_hef_path, audio_path):
    """Main evaluation function."""
    print("=" * 60)
    print("Whisper Tiny HEF Model Evaluation")
    print("=" * 60)
    
    # Load and preprocess audio
    print("\n[1] Loading audio file...")
    waveform, sr = load_and_preprocess_audio(audio_path)
    print(f"    Audio shape: {waveform.shape}, Sample rate: {sr}Hz")
    
    # Prepare encoder input
    print("\n[2] Preparing encoder input (mel-spectrogram)...")
    encoder_input = prepare_encoder_input(waveform, sr)
    print(f"    Encoder input shape: {encoder_input.shape}")
    
    # Load encoder model
    print("\n[3] Loading encoder model...")
    try:
        encoder_runner = ClientRunner(encoder_hef_path)
        print(f"    Encoder loaded successfully from: {encoder_hef_path}")
    except Exception as e:
        print(f"    Error loading encoder: {e}")
        return
    
    # Run encoder
    print("\n[4] Running encoder inference...")
    encoder_output, encoder_time = run_encoder(encoder_runner, encoder_input)
    
    # Load decoder model
    print("\n[5] Loading decoder model...")
    try:
        decoder_runner = ClientRunner(decoder_hef_path)
        print(f"    Decoder loaded successfully from: {decoder_hef_path}")
    except Exception as e:
        print(f"    Error loading decoder: {e}")
        return
    
    # Run decoder
    print("\n[6] Running decoder inference...")
    decoder_output, decoder_time = run_decoder(decoder_runner, encoder_output)
    
    # Print evaluation summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print(f"Encoder HEF: {encoder_hef_path}")
    print(f"Decoder HEF: {decoder_hef_path}")
    print(f"\nTiming Results:")
    print(f"  Encoder time: {encoder_time:.4f}s")
    print(f"  Decoder time: {decoder_time:.4f}s")
    print(f"  Total time: {encoder_time + decoder_time:.4f}s")
    print(f"\nModel Outputs:")
    print(f"  Encoder output shape: {encoder_output.shape}")
    print(f"  Decoder output shape: {decoder_output[0].shape}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Whisper Tiny HEF model")
    parser.add_argument("--encoder", required=True, help="Path to encoder HEF file")
    parser.add_argument("--decoder", required=True, help="Path to decoder HEF file")
    parser.add_argument("--audio", required=True, help="Path to audio WAV file")
    
    args = parser.parse_args()
    
    evaluate_model(args.encoder, args.decoder, args.audio)