import time
import numpy as np
from hailo_platform import (VDevice, HailoStreamInterface, ConfigureParams,
                            InferVStreams, InputVStreamParams, OutputVStreamParams)
import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from scipy.io import wavfile
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioChunker:
    """Handle audio loading and chunking for Whisper model"""
    
    def __init__(self, chunk_duration: float = 30.0, sample_rate: int = 16000):
        """
        Initialize audio chunker
        
        Args:
            chunk_duration: Duration of each chunk in seconds (Whisper uses 30s)
            sample_rate: Target sample rate (Whisper uses 16kHz)
        """
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.n_fft = 400
        self.hop_length = 160
        self.n_mels = 80
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate"""
        logger.info(f"Loading audio from {audio_path}")
        
        # Load audio with scipy.io.wavfile
        sr, audio = wavfile.read(audio_path)
        
        # Convert to float32 and normalize
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float32) - 128) / 128.0
        
        # Convert stereo to mono if necessary
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if necessary
        if sr != self.sample_rate:
            logger.info(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
            num_samples = int(len(audio) * self.sample_rate / sr)
            audio = signal.resample(audio, num_samples)
            sr = self.sample_rate
        
        logger.info(f"Audio loaded: duration={len(audio)/sr:.2f}s, sample_rate={sr}Hz")
        return audio, sr
    
    def chunk_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """
        Split audio into chunks of specified duration
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(self.chunk_duration * sr)
        chunks = []
        
        # Pad the last chunk if necessary
        total_samples = len(audio)
        num_chunks = int(np.ceil(total_samples / chunk_samples))
        
        for i in range(num_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, total_samples)
            
            chunk = audio[start:end]
            
            # Pad last chunk if necessary
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            
            chunks.append(chunk)
        
        logger.info(f"Audio split into {len(chunks)} chunks of {self.chunk_duration}s each")
        return chunks
    
    def audio_to_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert audio to mel spectrogram (Whisper input format)
        
        Args:
            audio: Audio signal
            
        Returns:
            Mel spectrogram
        """
        # Compute STFT
        f, t, Zxx = signal.stft(
            audio,
            fs=self.sample_rate,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            nfft=self.n_fft
        )
        
        # Get magnitude spectrogram
        mag_spec = np.abs(Zxx)
        power_spec = mag_spec ** 2
        
        # Create mel filterbank
        mel_basis = self._create_mel_filterbank()
        
        # Apply mel filterbank
        mel_spec = np.dot(mel_basis, power_spec)
        
        # Convert to log scale
        log_mel_spec = 10 * np.log10(np.maximum(mel_spec, 1e-10))
        
        # Normalize
        log_mel_spec = (log_mel_spec + 40) / 40
        
        return log_mel_spec
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """Create mel filterbank matrix"""
        # Frequency range
        fmin = 0
        fmax = 8000
        
        # Convert to mel scale
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        
        # Create mel points
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        
        # Convert back to Hz
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        # Create frequency bins
        n_fft_bins = self.n_fft // 2 + 1
        freq_bins = np.linspace(0, self.sample_rate / 2, n_fft_bins)
        
        # Create filterbank
        filterbank = np.zeros((self.n_mels, n_fft_bins))
        
        for i in range(self.n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            
            for j, freq in enumerate(freq_bins):
                if left <= freq <= center:
                    filterbank[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filterbank[i, j] = (right - freq) / (right - center)
        
        return filterbank
    
    def prepare_whisper_input(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Prepare audio chunk as Whisper encoder input
        
        Args:
            audio_chunk: Audio signal chunk
            
        Returns:
            Formatted input for Whisper encoder
        """
        mel_spec = self.audio_to_mel_spectrogram(audio_chunk)
        
        # Whisper expects shape: (batch_size, n_mels, time_steps)
        # Add batch dimension
        mel_spec = np.expand_dims(mel_spec, axis=0)
        
        return mel_spec.astype(np.float32)


class WhisperHailoEvaluator:
    def __init__(self, encoder_hef_path: str, decoder_hef_path: str):
        """
        Initialize Whisper model evaluator for Hailo NPU
        
        Args:
            encoder_hef_path: Path to encoder .hef file
            decoder_hef_path: Path to decoder .hef file
        """
        self.encoder_hef_path = encoder_hef_path
        self.decoder_hef_path = decoder_hef_path
        self.encoder_network = None
        self.decoder_network = None
        self.device = None
        self.audio_chunker = AudioChunker()
        self.metrics = {
            'model_loading': {},
            'audio_processing': {},
            'inference': {},
            'overall': {}
        }
    
    def load_models(self) -> Dict[str, float]:
        """Load encoder and decoder models and measure loading time"""
        logger.info("Starting model loading...")
        
        # Device initialization
        start_device = time.time()
        self.device = VDevice()
        device_time = time.time() - start_device
        logger.info(f"Device initialization: {device_time:.4f}s")
        
        # Load encoder
        start_encoder = time.time()
        encoder_params = ConfigureParams.create_from_hef(self.encoder_hef_path)
        self.encoder_network = self.device.configure(encoder_params)[0]
        encoder_time = time.time() - start_encoder
        logger.info(f"Encoder loading: {encoder_time:.4f}s")
        
        # Load decoder
        start_decoder = time.time()
        decoder_params = ConfigureParams.create_from_hef(self.decoder_hef_path)
        self.decoder_network = self.device.configure(decoder_params)[0]
        decoder_time = time.time() - start_decoder
        logger.info(f"Decoder loading: {decoder_time:.4f}s")
        
        total_loading_time = time.time() - start_device
        
        self.metrics['model_loading'] = {
            'device_init_time': device_time,
            'encoder_load_time': encoder_time,
            'decoder_load_time': decoder_time,
            'total_load_time': total_loading_time
        }
        
        logger.info(f"Total model loading time: {total_loading_time:.4f}s")
        return self.metrics['model_loading']
    
    def get_model_info(self):
        """Get information about the loaded models"""
        encoder_info = {
            'name': self.encoder_network.name,
            'inputs': [(inp.name, inp.shape) for inp in self.encoder_network.get_input_vstream_infos()],
            'outputs': [(out.name, out.shape) for out in self.encoder_network.get_output_vstream_infos()]
        }
        
        decoder_info = {
            'name': self.decoder_network.name,
            'inputs': [(inp.name, inp.shape) for inp in self.decoder_network.get_input_vstream_infos()],
            'outputs': [(out.name, out.shape) for out in self.decoder_network.get_output_vstream_infos()]
        }
        
        logger.info(f"Encoder inputs: {encoder_info['inputs']}")
        logger.info(f"Encoder outputs: {encoder_info['outputs']}")
        logger.info(f"Decoder inputs: {decoder_info['inputs']}")
        logger.info(f"Decoder outputs: {decoder_info['outputs']}")
        
        return encoder_info, decoder_info
    
    def process_audio_file(self, audio_path: str) -> List[np.ndarray]:
        """
        Load and process audio file into chunks
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of processed audio chunks ready for inference
        """
        start_time = time.time()
        
        # Load audio
        audio, sr = self.audio_chunker.load_audio(audio_path)
        load_time = time.time() - start_time
        
        # Chunk audio
        start_chunk = time.time()
        audio_chunks = self.audio_chunker.chunk_audio(audio, sr)
        chunk_time = time.time() - start_chunk
        
        # Convert to mel spectrograms
        start_mel = time.time()
        mel_chunks = [self.audio_chunker.prepare_whisper_input(chunk) 
                      for chunk in audio_chunks]
        mel_time = time.time() - start_mel
        
        total_processing_time = time.time() - start_time
        
        self.metrics['audio_processing'] = {
            'audio_duration': len(audio) / sr,
            'num_chunks': len(audio_chunks),
            'load_time': load_time,
            'chunk_time': chunk_time,
            'mel_conversion_time': mel_time,
            'total_processing_time': total_processing_time,
            'processing_per_chunk': total_processing_time / len(audio_chunks)
        }
        
        logger.info(f"Audio processing complete: {len(mel_chunks)} chunks in {total_processing_time:.4f}s")
        return mel_chunks
    
    def run_encoder_inference(self, mel_input: np.ndarray) -> Tuple[Dict, float]:
        """Run encoder inference and measure time"""
        start = time.time()
        
        with InferVStreams(self.encoder_network, 
                          InputVStreamParams.make_from_network_group(self.encoder_network),
                          OutputVStreamParams.make_from_network_group(self.encoder_network)) as infer_pipeline:
            
            # Get input layer name
            input_vstream_info = self.encoder_network.get_input_vstream_infos()[0]
            input_name = input_vstream_info.name
            
            # Prepare input dictionary
            input_dict = {input_name: mel_input}
            
            # Run inference
            output = infer_pipeline.infer(input_dict)
        
        inference_time = time.time() - start
        return output, inference_time
    
    def run_decoder_inference(self, encoder_output: Dict, max_tokens: int = 448) -> Tuple[List[int], float]:
        """
        Run decoder inference (autoregressive)
        
        Args:
            encoder_output: Output from encoder
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated token IDs and inference time
        """
        start = time.time()
        generated_tokens = []
        
        # Get decoder input/output info
        decoder_input_infos = self.decoder_network.get_input_vstream_infos()
        
        with InferVStreams(self.decoder_network,
                          InputVStreamParams.make_from_network_group(self.decoder_network),
                          OutputVStreamParams.make_from_network_group(self.decoder_network)) as infer_pipeline:
            
            # For simplicity, run decoder once (in practice, this would be autoregressive)
            # You'll need to adapt this based on your actual decoder input requirements
            decoder_inputs = {}
            
            for input_info in decoder_input_infos:
                input_name = input_info.name
                input_shape = input_info.shape
                
                # Use encoder output or create appropriate decoder input
                if 'encoder' in input_name.lower() or 'cross' in input_name.lower():
                    # This should be the encoder output
                    # Get the actual encoder output tensor
                    encoder_output_name = list(encoder_output.keys())[0]
                    decoder_inputs[input_name] = encoder_output[encoder_output_name]
                else:
                    # For token inputs or position embeddings, create dummy data
                    decoder_inputs[input_name] = np.zeros(input_shape, dtype=np.float32)
            
            # Run decoder inference
            output = infer_pipeline.infer(decoder_inputs)
            
            # Extract token predictions (you'll need to adapt this based on output format)
            output_name = list(output.keys())[0]
            logits = output[output_name]
            tokens = np.argmax(logits, axis=-1).flatten().tolist()
            generated_tokens.extend(tokens[:50])  # Take first 50 tokens as example
        
        inference_time = time.time() - start
        return generated_tokens, inference_time
    
    def translate_audio_file(self, audio_path: str) -> Dict:
        """
        Complete pipeline: load audio, chunk, and translate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Translation results and timing metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting translation of: {audio_path}")
        logger.info(f"{'='*70}\n")
        
        overall_start = time.time()
        
        # Process audio into chunks
        mel_chunks = self.process_audio_file(audio_path)
        
        # Initialize timing arrays
        encoder_times = []
        decoder_times = []
        chunk_total_times = []
        all_translations = []
        
        # Process each chunk
        for i, mel_chunk in enumerate(mel_chunks):
            logger.info(f"\nProcessing chunk {i+1}/{len(mel_chunks)}")
            
            chunk_start = time.time()
            
            # Run encoder
            encoder_output, encoder_time = self.run_encoder_inference(mel_chunk)
            encoder_times.append(encoder_time)
            logger.info(f"  Encoder: {encoder_time:.4f}s")
            
            # Run decoder
            tokens, decoder_time = self.run_decoder_inference(encoder_output)
            decoder_times.append(decoder_time)
            logger.info(f"  Decoder: {decoder_time:.4f}s")
            
            chunk_time = time.time() - chunk_start
            chunk_total_times.append(chunk_time)
            
            all_translations.append({
                'chunk_id': i,
                'tokens': tokens,
                'encoder_time': encoder_time,
                'decoder_time': decoder_time,
                'total_time': chunk_time
            })
            
            logger.info(f"  Chunk total: {chunk_time:.4f}s")
            logger.info(f"  Generated {len(tokens)} tokens")
        
        total_translation_time = time.time() - overall_start
        
        # Calculate statistics
        self.metrics['inference'] = {
            'num_chunks': len(mel_chunks),
            'encoder': {
                'mean': np.mean(encoder_times),
                'std': np.std(encoder_times),
                'min': np.min(encoder_times),
                'max': np.max(encoder_times),
                'total': np.sum(encoder_times),
                'all_times': encoder_times
            },
            'decoder': {
                'mean': np.mean(decoder_times),
                'std': np.std(decoder_times),
                'min': np.min(decoder_times),
                'max': np.max(decoder_times),
                'total': np.sum(decoder_times),
                'all_times': decoder_times
            },
            'per_chunk': {
                'mean': np.mean(chunk_total_times),
                'std': np.std(chunk_total_times),
                'min': np.min(chunk_total_times),
                'max': np.max(chunk_total_times),
                'all_times': chunk_total_times
            }
        }
        
        self.metrics['overall'] = {
            'total_translation_time': total_translation_time,
            'audio_duration': self.metrics['audio_processing']['audio_duration'],
            'real_time_factor': total_translation_time / self.metrics['audio_processing']['audio_duration']
        }
        
        return {
            'translations': all_translations,
            'metrics': self.metrics
        }
    
    def calculate_throughput(self):
        """Calculate throughput metrics"""
        if 'inference' not in self.metrics or not self.metrics['inference']:
            logger.warning("No inference metrics available.")
            return {}
        
        audio_duration = self.metrics['audio_processing']['audio_duration']
        total_time = self.metrics['overall']['total_translation_time']
        
        throughput = {
            'real_time_factor': self.metrics['overall']['real_time_factor'],
            'audio_duration': audio_duration,
            'total_processing_time': total_time,
            'chunks_per_second': self.metrics['inference']['num_chunks'] / total_time,
            'encoder_throughput': 1.0 / self.metrics['inference']['encoder']['mean'],
            'decoder_throughput': 1.0 / self.metrics['inference']['decoder']['mean']
        }
        
        self.metrics['throughput'] = throughput
        return throughput
    
    def print_summary(self):
        """Print a formatted summary of all metrics"""
        print("\n" + "="*70)
        print("HAILO NPU WHISPER TRANSLATION PERFORMANCE SUMMARY")
        print("="*70)
        
        # Model Loading
        print("\n📦 MODEL LOADING METRICS:")
        print("-" * 70)
        loading = self.metrics['model_loading']
        print(f"  Device Initialization:  {loading['device_init_time']:.4f}s")
        print(f"  Encoder Load Time:      {loading['encoder_load_time']:.4f}s")
        print(f"  Decoder Load Time:      {loading['decoder_load_time']:.4f}s")
        print(f"  Total Loading Time:     {loading['total_load_time']:.4f}s")
        
        # Audio Processing
        if 'audio_processing' in self.metrics:
            print("\n🎵 AUDIO PROCESSING METRICS:")
            print("-" * 70)
            audio = self.metrics['audio_processing']
            print(f"  Audio Duration:         {audio['audio_duration']:.2f}s")
            print(f"  Number of Chunks:       {audio['num_chunks']}")
            print(f"  Audio Load Time:        {audio['load_time']:.4f}s")
            print(f"  Chunking Time:          {audio['chunk_time']:.4f}s")
            print(f"  Mel Conversion Time:    {audio['mel_conversion_time']:.4f}s")
            print(f"  Total Processing Time:  {audio['total_processing_time']:.4f}s")
            print(f"  Time per Chunk:         {audio['processing_per_chunk']:.4f}s")
        
        # Inference
        if 'inference' in self.metrics and self.metrics['inference']:
            print("\n⚡ INFERENCE METRICS:")
            print("-" * 70)
            inf = self.metrics['inference']
            
            print(f"  Total Chunks Processed: {inf['num_chunks']}")
            
            print(f"\n  Encoder Inference (per chunk):")
            print(f"    Mean:  {inf['encoder']['mean']:.4f}s ± {inf['encoder']['std']:.4f}s")
            print(f"    Min:   {inf['encoder']['min']:.4f}s")
            print(f"    Max:   {inf['encoder']['max']:.4f}s")
            print(f"    Total: {inf['encoder']['total']:.4f}s")
            
            print(f"\n  Decoder Inference (per chunk):")
            print(f"    Mean:  {inf['decoder']['mean']:.4f}s ± {inf['decoder']['std']:.4f}s")
            print(f"    Min:   {inf['decoder']['min']:.4f}s")
            print(f"    Max:   {inf['decoder']['max']:.4f}s")
            print(f"    Total: {inf['decoder']['total']:.4f}s")
            
            print(f"\n  Complete Chunk Processing:")
            print(f"    Mean:  {inf['per_chunk']['mean']:.4f}s ± {inf['per_chunk']['std']:.4f}s")
            print(f"    Min:   {inf['per_chunk']['min']:.4f}s")
            print(f"    Max:   {inf['per_chunk']['max']:.4f}s")
        
        # Overall
        if 'overall' in self.metrics:
            print("\n🎯 OVERALL METRICS:")
            print("-" * 70)
            overall = self.metrics['overall']
            print(f"  Audio Duration:          {overall['audio_duration']:.2f}s")
            print(f"  Total Translation Time:  {overall['total_translation_time']:.2f}s")
            print(f"  Real-Time Factor:        {overall['real_time_factor']:.2f}x")
            if overall['real_time_factor'] < 1.0:
                print(f"  ✅ FASTER than real-time!")
            else:
                print(f"  ⚠️  SLOWER than real-time")
        
        # Throughput
        if 'throughput' in self.metrics:
            print("\n🚀 THROUGHPUT METRICS:")
            print("-" * 70)
            tp = self.metrics['throughput']
            print(f"  Chunks per Second:      {tp['chunks_per_second']:.2f}")
            print(f"  Encoder Throughput:     {tp['encoder_throughput']:.2f} chunks/sec")
            print(f"  Decoder Throughput:     {tp['decoder_throughput']:.2f} chunks/sec")
        
        print("\n" + "="*70)
    
    def save_metrics(self, output_path: str = "whisper_translation_metrics.json"):
        """Save metrics to JSON file"""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        metrics_serializable = convert_numpy(self.metrics)
        
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        if self.encoder_network:
            del self.encoder_network
        if self.decoder_network:
            del self.decoder_network
        if self.device:
            del self.device


def main():
    # Configuration
    ENCODER_HEF_PATH = "path/to/whisper_encoder.hef"
    DECODER_HEF_PATH = "path/to/whisper_decoder.hef"
    AUDIO_FILE_PATH = "path/to/audio.wav"  # Your input audio file
    OUTPUT_JSON = "whisper_translation_results.json"
    
    try:
        # Initialize evaluator
        evaluator = WhisperHailoEvaluator(ENCODER_HEF_PATH, DECODER_HEF_PATH)
        
        # Load models
        print("\n🔄 Loading models...")
        evaluator.load_models()
        
        # Get model information
        print("\n📊 Model Information:")
        encoder_info, decoder_info = evaluator.get_model_info()
        
        # Translate audio file
        print("\n🎤 Processing audio file...")
        results = evaluator.translate_audio_file(AUDIO_FILE_PATH)
        
        # Calculate throughput
        print("\n📈 Calculating throughput...")
        evaluator.calculate_throughput()
        
        # Print summary
        evaluator.print_summary()
        
        # Save metrics
        evaluator.save_metrics(OUTPUT_JSON)
        
        # Cleanup
        evaluator.cleanup()
        
        print(f"\n✅ Translation complete! Results saved to {OUTPUT_JSON}")
        
    except Exception as e:
        logger.error(f"Error during translation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
