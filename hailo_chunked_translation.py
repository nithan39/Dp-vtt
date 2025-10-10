"""Simple Hailo NPU Whisper Translation with Audio Chunking"""

import numpy as np
import wave
import time
import os
from app.hailo_whisper_pipeline import HailoWhisperPipeline
from common.preprocessing import preprocess
from app.whisper_hef_registry import HEF_REGISTRY


def load_audio(file_path):
    """Load WAV file and return audio data as numpy array."""
    with wave.open(file_path, 'rb') as wf:
        # Get audio parameters
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        
        # Read audio data
        audio_data = wf.readframes(n_frames)
        
        # Convert to numpy array
        if sampwidth == 2:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
        elif sampwidth == 4:
            audio_array = np.frombuffer(audio_data, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")
        
        # Convert stereo to mono if needed
        if n_channels == 2:
            audio_array = audio_array.reshape(-1, 2).mean(axis=1).astype(audio_array.dtype)
        
        # Convert to float32 and normalize to [-1, 1]
        audio_float = audio_array.astype(np.float32) / np.iinfo(audio_array.dtype).max
        
        return audio_float, framerate


def resample_audio(audio, orig_sr, target_sr=16000):
    """Simple resampling using linear interpolation."""
    if orig_sr == target_sr:
        return audio
    
    # Calculate new length
    duration = len(audio) / orig_sr
    new_length = int(duration * target_sr)
    
    # Create new time points
    old_indices = np.linspace(0, len(audio) - 1, len(audio))
    new_indices = np.linspace(0, len(audio) - 1, new_length)
    
    # Interpolate
    resampled = np.interp(new_indices, old_indices, audio)
    return resampled


def chunk_audio(audio, chunk_duration=30, sample_rate=16000, overlap=0):
    """
    Split audio into chunks with optional overlap.
    
    Args:
        audio: Audio array
        chunk_duration: Duration of each chunk in seconds
        sample_rate: Sample rate of audio
        overlap: Overlap duration in seconds
    
    Returns:
        List of audio chunks
    """
    chunk_samples = int(chunk_duration * sample_rate)
    overlap_samples = int(overlap * sample_rate)
    step = chunk_samples - overlap_samples
    
    chunks = []
    for start in range(0, len(audio), step):
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]
        
        # Pad last chunk if needed
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
        
        chunks.append(chunk)
        
        # Break if we've reached the end
        if end >= len(audio):
            break
    
    return chunks


def get_hef_path(model_variant: str, hw_arch: str, component: str) -> str:
    """
    Method to retrieve HEF path.

    Args:
        model_variant (str): e.g. "tiny", "base"
        hw_arch (str): e.g. "hailo8", "hailo8l"
        component (str): "encoder" or "decoder"

    Returns:
        str: Absolute path to the requested HEF file.
    """
    try:
        hef_path = HEF_REGISTRY[model_variant][hw_arch][component]
    except KeyError as e:
        raise FileNotFoundError(
            f"HEF not available for model '{model_variant}' on hardware '{hw_arch}'."
        ) from e

    if not os.path.exists(hef_path):
        raise FileNotFoundError(
            f"HEF file not found at: {hef_path}\n"
            f"Please run: python3 ./download_resources.py --hw-arch {hw_arch}"
        )
    return hef_path


def transcribe_chunks(encoder_path, decoder_path, audio_file, variant="tiny", 
                     chunk_duration=30, overlap=1):
    """
    Perform chunked inference on audio file using Hailo NPU.
    
    Args:
        encoder_path: Path to encoder HEF file
        decoder_path: Path to decoder HEF file
        audio_file: Path to WAV audio file
        variant: Whisper variant ('tiny', 'base', 'tiny.en')
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
    
    Returns:
        Full transcription/translation text
    """
    # Load model
    print(f"Loading Hailo Whisper model (variant: {variant})...")
    whisper_hailo = HailoWhisperPipeline(encoder_path, decoder_path, variant)
    
    # Load and preprocess audio
    print(f"Loading audio from {audio_file}...")
    audio, sample_rate = load_audio(audio_file)
    
    # Resample to 16kHz if needed (Whisper requirement)
    if sample_rate != 16000:
        print(f"Resampling from {sample_rate}Hz to 16000Hz...")
        audio = resample_audio(audio, sample_rate, 16000)
        sample_rate = 16000
    
    # Split into chunks
    print(f"Splitting audio into {chunk_duration}s chunks with {overlap}s overlap...")
    chunks = chunk_audio(audio, chunk_duration, sample_rate, overlap)
    print(f"Created {len(chunks)} chunks")
    
    # Preprocess parameters
    chunk_length = 10 if "tiny" in variant else 5
    is_nhwc = True
    
    # Transcribe each chunk
    full_transcription = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        # Convert to mel spectrograms
        mel_spectrograms = preprocess(
            chunk,
            is_nhwc=is_nhwc,
            chunk_length=chunk_length,
            chunk_offset=0
        )
        
        # Process each mel spectrogram
        for mel in mel_spectrograms:
            whisper_hailo.send_data(mel)
            time.sleep(0.1)
            transcription = whisper_hailo.get_transcription()
            
            # Extract text
            chunk_text = transcription.strip()
            if chunk_text:
                full_transcription.append(chunk_text)
                print(f"Chunk {i+1}: {chunk_text}")
    
    # Stop pipeline
    whisper_hailo.stop()
    
    # Combine all transcriptions
    final_text = " ".join(full_transcription)
    return final_text


if __name__ == "__main__":
    # Configuration
    HW_ARCH = "hailo8"           # Hardware architecture: hailo8, hailo8l, hailo10h
    VARIANT = "tiny"              # Model variant: tiny, base, tiny.en
    AUDIO_FILE = "sample.wav"     # Path to your audio file
    CHUNK_DURATION = 30           # Chunk duration in seconds
    OVERLAP = 1                   # Overlap between chunks in seconds
    
    # Get HEF model paths
    try:
        ENCODER_PATH = get_hef_path(VARIANT, HW_ARCH, "encoder")
        DECODER_PATH = get_hef_path(VARIANT, HW_ARCH, "decoder")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Perform transcription
    try:
        transcription = transcribe_chunks(
            encoder_path=ENCODER_PATH,
            decoder_path=DECODER_PATH,
            audio_file=AUDIO_FILE,
            variant=VARIANT,
            chunk_duration=CHUNK_DURATION,
            overlap=OVERLAP
        )
        
        print("\n" + "="*50)
        print("FULL TRANSCRIPTION:")
        print("="*50)
        print(transcription)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
