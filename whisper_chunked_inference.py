import numpy as np
import wave
from pywhispercpp.model import Model

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

def transcribe_chunks(model_path, audio_file, chunk_duration=30, overlap=0):
    """
    Perform chunked inference on audio file.
    
    Args:
        model_path: Path to GGML model file (e.g., 'ggml-tiny.bin')
        audio_file: Path to WAV audio file
        chunk_duration: Duration of each chunk in seconds
        overlap: Overlap between chunks in seconds
    
    Returns:
        Full transcription text
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = Model(model_path)
    
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
    
    # Transcribe each chunk
    full_transcription = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        
        # Transcribe chunk
        result = model.transcribe(chunk)
        
        # Extract text
        chunk_text = result.strip()
        if chunk_text:
            full_transcription.append(chunk_text)
            print(f"Chunk {i+1}: {chunk_text}")
    
    # Combine all transcriptions
    final_text = " ".join(full_transcription)
    return final_text

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "ggml-tiny.bin"  # Path to your GGML model
    AUDIO_FILE = "sample.wav"      # Path to your audio file
    CHUNK_DURATION = 30            # Chunk duration in seconds
    OVERLAP = 1                    # Overlap between chunks in seconds
    
    # Perform transcription
    try:
        transcription = transcribe_chunks(
            model_path=MODEL_PATH,
            audio_file=AUDIO_FILE,
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
