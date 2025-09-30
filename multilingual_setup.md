# Multilingual Real-Time Voice Translator

Automatically detects and translates mixed-language speech to English in real-time!

## 🌟 Features

- ✅ **Automatic Language Detection** - Detects 20+ languages automatically
- ✅ **Mixed Language Support** - Handles English → Tamil → Chinese seamlessly
- ✅ **Real-time Translation** - Translates to English sentence-by-sentence
- ✅ **Language Statistics** - Tracks which languages were spoken
- ✅ **100% Local** - No internet required after setup

## 🎯 How It Works

```
Audio Input (Mixed Languages)
    ↓
Voice Activity Detection
    ↓
Whisper.cpp (Auto-detect Language)
    ↓
Language Identified (en/ta/zh/es/etc.)
    ↓
Sentence Segmentation
    ↓
Translation to English (if not English)
    ↓
Display English Text
```

## 📦 Quick Setup

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git portaudio19-dev
sudo apt-get install -y python3-pip
```

**macOS:**
```bash
brew install portaudio cmake python3
```

### 2. Install Python Dependencies

```bash
pip3 install transformers torch sentencepiece protobuf
```

### 3. Build Whisper.cpp and Download Multilingual Model

```bash
# Clone and build whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make

# Download MULTILINGUAL model (NOT .en model!)
bash ./models/download-ggml-model.sh base

# Verify model
ls -lh ./models/ggml-base.bin
cd ..
```

### 4. Build the Translator

```bash
# Using provided Makefile
make

# OR manually
g++ -std=c++17 -O3 -o multilingual_translator multilingual_translator.cpp \
    -I./whisper.cpp \
    ./whisper.cpp/libwhisper.a \
    -lportaudio -pthread -lm
```

### 5. Run

```bash
./multilingual_translator ./whisper.cpp/models/ggml-base.bin
```

## 🎬 Usage Example

```bash
# Start the translator
./multilingual_translator ./whisper.cpp/models/ggml-base.bin

# Now speak in different languages:
# "Hello, how are you?"  (English)
# "நான் நன்றாக இருக்கிறேன்"  (Tamil)
# "我很好，谢谢"  (Chinese)

# Output:
# 🌐 Detected Language: English (en)
# ✅ English: Hello, how are you?
#
# 🌐 Detected Language: Tamil (ta)
# ✅ English: I am doing well
#
# 🌐 Detected Language: Chinese (zh)
# ✅ English: I'm fine, thank you
```

## 🌍 Supported Languages

The system automatically detects and translates from these languages to English:

| Language | Code | Translation Model |
|----------|------|-------------------|
| English | en | (no translation needed) |
| Chinese | zh | Helsinki-NLP/opus-mt-zh-en |
| Tamil | ta | Helsinki-NLP/opus-mt-mul-en |
| Spanish | es | Helsinki-NLP/opus-mt-es-en |
| French | fr | Helsinki-NLP/opus-mt-fr-en |
| German | de | Helsinki-NLP/opus-mt-de-en |
| Japanese | ja | Helsinki-NLP/opus-mt-jap-en |
| Korean | ko | Helsinki-NLP/opus-mt-ko-en |
| Hindi | hi | Helsinki-NLP/opus-mt-hi-en |
| Arabic | ar | Helsinki-NLP/opus-mt-ar-en |
| Russian | ru | Helsinki-NLP/opus-mt-ru-en |
| Portuguese | pt | Helsinki-NLP/opus-mt-roa-en |
| Italian | it | Helsinki-NLP/opus-mt-it-en |
| Dutch | nl | Helsinki-NLP/opus-mt-nl-en |
| Turkish | tr | Helsinki-NLP/opus-mt-tr-en |
| Polish | pl | Helsinki-NLP/opus-mt-pl-en |
| Vietnamese | vi | Helsinki-NLP/opus-mt-vi-en |
| Thai | th | Helsinki-NLP/opus-mt-th-en |
| Indonesian | id | Helsinki-NLP/opus-mt-id-en |
| Malay | ms | Helsinki-NLP/opus-mt-ms-en |

## 📊 Output Format

The translator displays:

1. **Detected Language** - Shows which language was identified
2. **Original Text** - The transcribed text in original language
3. **English Translation** - Translated sentence(s) in English
4. **Language Statistics** - Summary of all languages detected

Example output:
```
🗣️  Speech detected...

📝 Transcribing with language detection...

🌐 Detected Language: Spanish (es)
💬 Original: Hola, ¿cómo estás? Espero que estés bien.

🔄 Translating to English...
⏳ Translating sentence 1/2...
✅ English [1/2]: Hello, how are you?
⏳ Translating sentence 2/2...
✅ English [2/2]: I hope you are well.
============================================================

📊 Language Statistics:
----------------------------------------
  English (en): 5 segments
  Spanish (es): 3 segments
  Chinese (zh): 2 segments
  Tamil (ta): 1 segments
----------------------------------------
```

## 🎛️ Configuration

### Adjust Voice Activity Detection

More sensitive (detects quieter speech):
```cpp
MultilingualRealtimeTranslator translator(model_path, 0.005f);
```

Less sensitive (ignores background noise):
```cpp
MultilingualRealtimeTranslator translator(model_path, 0.02f);
```

### Change Silence Duration

Edit in `multilingual_translator.cpp`:
```cpp
#define SPEECH_PAD_MS 500  // Milliseconds of silence before processing
```

Lower value = faster processing but may cut off speech
Higher value = more complete sentences but slower

## 🚀 Performance Tips

### 1. Choose the Right Model

| Model | Size | Speed | Accuracy | Languages |
|-------|------|-------|----------|-----------|
| tiny | 75 MB | Fastest | Good | 99 |
| base | 142 MB | **Recommended** | Very Good | 99 |
| small | 466 MB | Fast | Excellent | 99 |
| medium | 1.5 GB | Medium | Best | 99 |
| large | 2.9 GB | Slow | Best | 99 |

Download different models:
```bash
cd whisper.cpp
bash ./models/download-ggml-model.sh tiny    # Fastest
bash ./models/download-ggml-model.sh base    # Recommended
bash ./models/download-ggml-model.sh small   # Better quality
```

### 2. First Run Optimization

**Important:** The first time you detect a new language, Python will download the translation model (~300MB). This is a **one-time download** and cached locally.

Pre-download translation models:
```bash
python3 -c "from transformers import MarianMTModel, MarianTokenizer; \
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-zh-en'); \
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ta-en')"
```

### 3. Hardware Acceleration

Use GPU for faster translation:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔧 Troubleshooting

### "Failed to load Whisper model"

❌ **Wrong:** Using English-only model
```bash
./multilingual_translator ./models/ggml-base.en.bin
```

✅ **Correct:** Using multilingual model
```bash
./multilingual_translator ./models/ggml-base.bin
```

### "Translation failed to execute"

Make sure Python dependencies are installed:
```bash
pip3 install transformers torch sentencepiece
```

### Translation is slow

**First run:** Models are being downloaded (one-time only)
**Subsequent runs:** Use smaller Whisper model or enable GPU

### Language not detected correctly

Use a