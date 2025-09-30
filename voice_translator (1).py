#include "whisper.h"
#include <portaudio.h>
#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <map>
#include <set>
#include <cstring>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <sstream>
#include <regex>

// Audio configuration
#define SAMPLE_RATE 16000
#define FRAMES_PER_BUFFER 1024
#define NUM_CHANNELS 1
#define SILENCE_THRESHOLD 0.01f
#define SPEECH_PAD_MS 500

// Language detection confidence threshold
#define LANG_DETECTION_CONFIDENCE 0.3f

// Supported languages map
const std::map<std::string, std::string> LANGUAGE_NAMES = {
    {"en", "English"},
    {"zh", "Chinese"},
    {"ta", "Tamil"},
    {"es", "Spanish"},
    {"fr", "French"},
    {"de", "German"},
    {"ja", "Japanese"},
    {"ko", "Korean"},
    {"hi", "Hindi"},
    {"ar", "Arabic"},
    {"ru", "Russian"},
    {"pt", "Portuguese"},
    {"it", "Italian"},
    {"nl", "Dutch"},
    {"tr", "Turkish"},
    {"pl", "Polish"},
    {"vi", "Vietnamese"},
    {"th", "Thai"},
    {"id", "Indonesian"},
    {"ms", "Malay"}
};

// Translation mapping for Python script (X -> en)
std::string get_translation_model(const std::string& lang) {
    static const std::map<std::string, std::string> models = {
        {"zh", "Helsinki-NLP/opus-mt-zh-en"},
        {"ta", "Helsinki-NLP/opus-mt-mul-en"},
        {"es", "Helsinki-NLP/opus-mt-es-en"},
        {"fr", "Helsinki-NLP/opus-mt-fr-en"},
        {"de", "Helsinki-NLP/opus-mt-de-en"},
        {"ja", "Helsinki-NLP/opus-mt-jap-en"},
        {"ko", "Helsinki-NLP/opus-mt-ko-en"},
        {"hi", "Helsinki-NLP/opus-mt-hi-en"},
        {"ar", "Helsinki-NLP/opus-mt-ar-en"},
        {"ru", "Helsinki-NLP/opus-mt-ru-en"},
        {"pt", "Helsinki-NLP/opus-mt-roa-en"},
        {"it", "Helsinki-NLP/opus-mt-it-en"},
        {"nl", "Helsinki-NLP/opus-mt-nl-en"},
        {"tr", "Helsinki-NLP/opus-mt-tr-en"},
        {"pl", "Helsinki-NLP/opus-mt-pl-en"},
        {"vi", "Helsinki-NLP/opus-mt-vi-en"},
        {"th", "Helsinki-NLP/opus-mt-th-en"},
        {"id", "Helsinki-NLP/opus-mt-id-en"},
        {"ms", "Helsinki-NLP/opus-mt-ms-en"}
    };
    
    auto it = models.find(lang);
    if (it != models.end()) {
        return it->second;
    }
    return "Helsinki-NLP/opus-mt-mul-en"; // Multilingual fallback
}

// Execute Python script for translation
std::string translate_to_english(const std::string& text, const std::string& source_lang) {
    if (source_lang == "en") {
        return text; // Already English
    }
    
    // Escape quotes in text
    std::string escaped_text = text;
    size_t pos = 0;
    while ((pos = escaped_text.find("\"", pos)) != std::string::npos) {
        escaped_text.replace(pos, 1, "\\\"");
        pos += 2;
    }
    
    std::string model = get_translation_model(source_lang);
    std::string cmd = "python3 -c 'from transformers import MarianMTModel, MarianTokenizer; "
                     "import sys; "
                     "model = MarianMTModel.from_pretrained(\"" + model + "\"); "
                     "tokenizer = MarianTokenizer.from_pretrained(\"" + model + "\"); "
                     "text = \"" + escaped_text + "\"; "
                     "inputs = tokenizer(text, return_tensors=\"pt\", padding=True); "
                     "translated = model.generate(**inputs); "
                     "print(tokenizer.decode(translated[0], skip_special_tokens=True), end=\"\")'";
    
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "❌ Translation failed to execute" << std::endl;
        return text;
    }
    
    char buffer[4096];
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    pclose(pipe);
    
    // Trim whitespace
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    result.erase(result.find_last_not_of(" \t\n\r") + 1);
    
    return result.empty() ? text : result;
}

// Split text into sentences
std::vector<std::string> split_sentences(const std::string& text) {
    std::vector<std::string> sentences;
    std::regex sentence_regex(R"([^.!?。！？]+[.!?。！？]+)");
    
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), sentence_regex);
    auto words_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::string sentence = (*i).str();
        sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
        sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);
        if (!sentence.empty()) {
            sentences.push_back(sentence);
        }
    }
    
    if (sentences.empty() && !text.empty()) {
        sentences.push_back(text);
    }
    
    return sentences;
}

struct TranscriptionResult {
    std::string text;
    std::string language;
    float confidence;
};

class MultilingualRealtimeTranslator {
private:
    whisper_context* ctx;
    PaStream* stream;
    
    std::queue<std::vector<float>> audio_queue;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    
    std::atomic<bool> running;
    std::thread processing_thread;
    
    std::vector<float> audio_buffer;
    std::string last_transcription;
    
    float vad_threshold;
    int silence_counter;
    bool is_speaking;
    
    // Language tracking
    std::map<std::string, int> language_stats;
    std::string last_detected_lang;
    
public:
    MultilingualRealtimeTranslator(const std::string& model_path, 
                                   float vad_thresh = SILENCE_THRESHOLD)
        : running(false), vad_threshold(vad_thresh), 
          silence_counter(0), is_speaking(false),
          last_detected_lang("unknown") {
        
        // Initialize Whisper with multilingual model
        std::cout << "Loading Whisper multilingual model..." << std::endl;
        struct whisper_context_params cparams = whisper_context_default_params();
        ctx = whisper_init_from_file_with_params(model_path.c_str(), cparams);
        
        if (!ctx) {
            throw std::runtime_error("Failed to load Whisper model. Please use a multilingual model (e.g., ggml-base.bin)");
        }
        
        std::cout << "✓ Whisper multilingual model loaded" << std::endl;
        
        // Initialize PortAudio
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            throw std::runtime_error("PortAudio initialization failed");
        }
        
        std::cout << "✓ Audio system initialized" << std::endl;
        std::cout << "✓ Automatic language detection enabled" << std::endl;
        std::cout << "✓ All languages will be translated to English" << std::endl;
    }
    
    ~MultilingualRealtimeTranslator() {
        stop();
        if (ctx) {
            whisper_free(ctx);
        }
        Pa_Terminate();
    }
    
    static int audio_callback(const void* input, void* output,
                            unsigned long frameCount,
                            const PaStreamCallbackTimeInfo* timeInfo,
                            PaStreamCallbackFlags statusFlags,
                            void* userData) {
        
        MultilingualRealtimeTranslator* translator = static_cast<MultilingualRealtimeTranslator*>(userData);
        const float* in = static_cast<const float*>(input);
        
        // Voice Activity Detection
        float energy = 0.0f;
        for (unsigned long i = 0; i < frameCount; i++) {
            energy += std::abs(in[i]);
        }
        energy /= frameCount;
        
        bool has_voice = energy > translator->vad_threshold;
        
        if (has_voice) {
            if (!translator->is_speaking) {
                std::cout << "\n🗣️  Speech detected..." << std::endl;
                translator->is_speaking = true;
            }
            translator->audio_buffer.insert(translator->audio_buffer.end(), in, in + frameCount);
            translator->silence_counter = 0;
        } else {
            translator->silence_counter++;
            
            if (translator->is_speaking) {
                translator->audio_buffer.insert(translator->audio_buffer.end(), in, in + frameCount);
                
                // End of speech (500ms silence)
                int silence_frames = (SAMPLE_RATE * SPEECH_PAD_MS) / (1000 * frameCount);
                if (translator->silence_counter > silence_frames) {
                    if (!translator->audio_buffer.empty()) {
                        std::lock_guard<std::mutex> lock(translator->queue_mutex);
                        translator->audio_queue.push(translator->audio_buffer);
                        translator->queue_cv.notify_one();
                        translator->audio_buffer.clear();
                    }
                    translator->is_speaking = false;
                    translator->silence_counter = 0;
                }
            }
        }
        
        return paContinue;
    }
    
    TranscriptionResult transcribe_with_language_detection(const std::vector<float>& audio) {
        TranscriptionResult result = {"", "unknown", 0.0f};
        
        if (audio.empty()) return result;
        
        // Configure Whisper for automatic language detection
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        wparams.language = nullptr; // Auto-detect language
        wparams.detect_language = true;
        wparams.translate = false; // We'll translate separately
        wparams.print_realtime = false;
        wparams.print_progress = false;
        wparams.print_timestamps = false;
        wparams.n_threads = 4;
        wparams.single_segment = false;
        
        // Run whisper
        if (whisper_full(ctx, wparams, audio.data(), audio.size()) != 0) {
            std::cerr << "❌ Whisper transcription failed" << std::endl;
            return result;
        }
        
        // Get detected language
        const char* detected_lang = whisper_lang_str(whisper_full_lang_id(ctx));
        result.language = detected_lang;
        
        // Get transcription
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; i++) {
            const char* text = whisper_full_get_segment_text(ctx, i);
            result.text += text;
        }
        
        // Trim whitespace
        result.text.erase(0, result.text.find_first_not_of(" \t\n\r"));
        result.text.erase(result.text.find_last_not_of(" \t\n\r") + 1);
        
        return result;
    }
    
    void display_language_stats() {
        if (language_stats.empty()) return;
        
        std::cout << "\n📊 Language Statistics:" << std::endl;
        std::cout << std::string(40, '-') << std::endl;
        
        // Sort by count
        std::vector<std::pair<std::string, int>> sorted_stats(
            language_stats.begin(), language_stats.end()
        );
        std::sort(sorted_stats.begin(), sorted_stats.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        for (const auto& stat : sorted_stats) {
            auto it = LANGUAGE_NAMES.find(stat.first);
            std::string lang_name = (it != LANGUAGE_NAMES.end()) ? it->second : stat.first;
            std::cout << "  " << lang_name << " (" << stat.first << "): " 
                     << stat.second << " segments" << std::endl;
        }
        std::cout << std::string(40, '-') << std::endl;
    }
    
    void processing_loop() {
        std::cout << "🔄 Processing thread started" << std::endl;
        
        while (running) {
            std::vector<float> audio_chunk;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                queue_cv.wait_for(lock, std::chrono::milliseconds(100), 
                    [this] { return !audio_queue.empty() || !running; });
                
                if (!running) break;
                
                if (!audio_queue.empty()) {
                    audio_chunk = audio_queue.front();
                    audio_queue.pop();
                }
            }
            
            if (!audio_chunk.empty()) {
                std::cout << "\n📝 Transcribing with language detection..." << std::endl;
                
                TranscriptionResult trans_result = transcribe_with_language_detection(audio_chunk);
                
                if (!trans_result.text.empty()) {
                    // Update language statistics
                    language_stats[trans_result.language]++;
                    
                    // Get language name
                    auto it = LANGUAGE_NAMES.find(trans_result.language);
                    std::string lang_name = (it != LANGUAGE_NAMES.end()) ? it->second : trans_result.language;
                    
                    std::cout << "\n🌐 Detected Language: " << lang_name 
                             << " (" << trans_result.language << ")" << std::endl;
                    std::cout << "💬 Original: " << trans_result.text << std::endl;
                    
                    // Split into sentences
                    std::vector<std::string> sentences = split_sentences(trans_result.text);
                    if (sentences.empty()) {
                        sentences.push_back(trans_result.text);
                    }
                    
                    // Translate each sentence to English
                    std::cout << "\n🔄 Translating to English..." << std::endl;
                    
                    for (size_t i = 0; i < sentences.size(); i++) {
                        const auto& sentence = sentences[i];
                        if (!sentence.empty()) {
                            if (trans_result.language == "en") {
                                std::cout << "✅ English [" << (i+1) << "/" << sentences.size() 
                                         << "]: " << sentence << std::endl;
                            } else {
                                std::cout << "⏳ Translating sentence " << (i+1) << "/" 
                                         << sentences.size() << "..." << std::endl;
                                
                                std::string translation = translate_to_english(sentence, trans_result.language);
                                
                                if (!translation.empty()) {
                                    std::cout << "✅ English [" << (i+1) << "/" << sentences.size() 
                                             << "]: " << translation << std::endl;
                                }
                            }
                        }
                    }
                    
                    std::cout << std::string(60, '=') << std::endl;
                    
                    last_transcription = trans_result.text;
                    last_detected_lang = trans_result.language;
                } else {
                    std::cout << "❌ No speech detected in audio." << std::endl;
                }
            }
        }
        
        // Display final statistics
        display_language_stats();
        
        std::cout << "🔄 Processing thread stopped" << std::endl;
    }
    
    void start() {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "🌍 MULTILINGUAL REAL-TIME TRANSLATOR" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "✓ Automatic language detection (20+ languages)" << std::endl;
        std::cout << "✓ Real-time translation to English" << std::endl;
        std::cout << "✓ Sentence-by-sentence processing" << std::endl;
        std::cout << "✓ Supports mixed-language conversations" << std::endl;
        std::cout << "✓ Press Ctrl+C to stop" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        std::cout << "\n📋 Supported Languages:" << std::endl;
        std::cout << "English, Chinese, Tamil, Spanish, French, German," << std::endl;
        std::cout << "Japanese, Korean, Hindi, Arabic, Russian, Portuguese," << std::endl;
        std::cout << "Italian, Dutch, Turkish, Polish, Vietnamese, Thai," << std::endl;
        std::cout << "Indonesian, Malay, and more..." << std::endl;
        std::cout << std::string(60, '=') << std::endl << std::endl;
        
        running = true;
        
        // Start processing thread
        processing_thread = std::thread(&MultilingualRealtimeTranslator::processing_loop, this);
        
        // Open audio stream
        PaStreamParameters inputParameters;
        inputParameters.device = Pa_GetDefaultInputDevice();
        inputParameters.channelCount = NUM_CHANNELS;
        inputParameters.sampleFormat = paFloat32;
        inputParameters.suggestedLatency = Pa_GetDeviceInfo(inputParameters.device)->defaultLowInputLatency;
        inputParameters.hostApiSpecificStreamInfo = nullptr;
        
        PaError err = Pa_OpenStream(&stream, &inputParameters, nullptr,
                                   SAMPLE_RATE, FRAMES_PER_BUFFER,
                                   paClipOff, audio_callback, this);
        
        if (err != paNoError) {
            throw std::runtime_error("Failed to open audio stream");
        }
        
        err = Pa_StartStream(stream);
        if (err != paNoError) {
            throw std::runtime_error("Failed to start audio stream");
        }
        
        std::cout << "🎤 Listening... Speak in any language!" << std::endl;
    }
    
    void stop() {
        if (!running) return;
        
        std::cout << "\n⏹️  Stopping..." << std::endl;
        running = false;
        
        queue_cv.notify_all();
        
        if (processing_thread.joinable()) {
            processing_thread.join();
        }
        
        if (stream) {
            Pa_StopStream(stream);
            Pa_CloseStream(stream);
            stream = nullptr;
        }
        
        std::cout << "✅ Translator stopped" << std::endl;
    }
};

int main(int argc, char** argv) {
    std::cout << "🌍 Multilingual Real-Time Voice Translator" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <multilingual_model_path>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " ./models/ggml-base.bin" << std::endl;
        std::cerr << "\n⚠️  Important: Use a MULTILINGUAL model (not .en model)" << std::endl;
        std::cerr << "Download with:" << std::endl;
        std::cerr << "  cd whisper.cpp" << std::endl;
        std::cerr << "  bash ./models/download-ggml-model.sh base" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    // Validate model is not English-only
    if (model_path.find(".en.bin") != std::string::npos) {
        std::cerr << "❌ Error: This is an English-only model!" << std::endl;
        std::cerr << "Please use a multilingual model (without .en)" << std::endl;
        std::cerr << "Download: bash whisper.cpp/models/download-ggml-model.sh base" << std::endl;
        return 1;
    }
    
    try {
        MultilingualRealtimeTranslator translator(model_path);
        translator.start();
        
        // Keep running until Ctrl+C
        std::cout << "\nPress Enter to stop and see statistics..." << std::endl;
        std::cin.get();
        
        translator.stop();
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
