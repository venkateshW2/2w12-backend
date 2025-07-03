#!/bin/bash
# Selective Essentia Model Downloader for 2W12 Sound Tools
# Choose which model categories to download

echo "🎵 2W12 Sound Tools - Selective Model Downloader"
echo "==============================================="
echo ""
echo "Choose which model categories to download:"
echo "1. ✅ Core Models (Key/Genre/Mood/Danceability) - ~80MB"
echo "2. ✅ Instrument Classification (Piano/Guitar/Drums/Voice etc.) - ~120MB"  
echo "3. ✅ Audio Quality & Production (Quality/Dynamics/Reverb) - ~30MB"
echo "4. ✅ Musicality & Style (Musicality/Decades/Electronic vs Acoustic) - ~45MB"
echo "5. ⚡ Advanced Emotions (Arousal/Valence/Engagement) - ~45MB"
echo "6. 📦 ALL MODELS - ~260MB total"
echo "7. 🎯 RECOMMENDED SET (Core + Instruments + Quality + Musicality) - ~200MB"
echo ""

# Function to download core models
download_core_models() {
    echo "🎯 Downloading Core Analysis Models (~80MB)..."
    echo "--------------------------------------------"
    
    echo "📥 Key detection model..."
    wget -q --show-progress https://essentia.upf.edu/models/discogs-effnet/discogs-effnet-bs64-1.pb -O models/discogs-effnet-key.pb
    
    echo "📥 Genre classification model..."
    wget -q --show-progress https://essentia.upf.edu/models/genre_discogs400/genre_discogs400-discogs-effnet-1.pb -O models/discogs-effnet-genres.pb
    
    echo "📥 Mood detection model..."
    wget -q --show-progress https://essentia.upf.edu/models/mood_acoustic/mood_acoustic-discogs-effnet-1.pb -O models/mood_acoustic.pb
    
    echo "📥 Danceability model..."
    wget -q --show-progress https://essentia.upf.edu/models/danceability/danceability-discogs-effnet-1.pb -O models/danceability.pb
    
    echo "✅ Core models downloaded!"
}

# Function to download instrument models
download_instrument_models() {
    echo "🎺 Downloading Instrument Classification Models (~120MB)..."
    echo "--------------------------------------------------------"
    
    echo "📥 Multi-instrument recognition..."
    wget -q --show-progress https://essentia.upf.edu/models/classifiers/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb -O models/instrument_recognition.pb
    
    echo "📥 Piano detection..."
    wget -q --show-progress https://essentia.upf.edu/models/classifiers/mtg_jamendo_instrument/piano-discogs-effnet-1.pb -O models/piano_detection.pb
    
    echo "📥 Guitar detection..."
    wget -q --show-progress https://essentia.upf.edu/models/classifiers/mtg_jamendo_instrument/guitar-discogs-effnet-1.pb -O models/guitar_detection.pb
    
    echo "📥 Violin detection..."
    wget -q --show-progress https://essentia.upf.edu/models/classifiers/mtg_jamendo_instrument/violin-discogs-effnet-1.pb -O models/violin_detection.pb
    
    echo "📥 Drums detection..."
    wget -q --show-progress https://essentia.upf.edu/models/classifiers/mtg_jamendo_instrument/drums-discogs-effnet-1.pb -O models/drums_detection.pb
    
    echo "📥 Voice detection..."
    wget -q --show-progress https://essentia.upf.edu/models/classifiers/mtg_jamendo_instrument/voice-discogs-effnet-1.pb -O models/voice_detection.pb
    
    echo "✅ Instrument models downloaded!"
}

# Function to download quality models
download_quality_models() {
    echo "🎚️ Downloading Audio Quality & Production Models (~30MB)..."
    echo "---------------------------------------------------------"
    
    echo "📥 Audio quality assessment..."
    wget -q --show-progress https://essentia.upf.edu/models/quality/quality-discogs-effnet-1.pb -O models/audio_quality.pb
    
    echo "📥 Dynamic range..."
    wget -q --show-progress https://essentia.upf.edu/models/dynamics/dynamics-discogs-effnet-1.pb -O models/dynamics.pb
    
    echo "📥 Reverb detection..."
    wget -q --show-progress https://essentia.upf.edu/models/reverb/reverb-discogs-effnet-1.pb -O models/reverb.pb
    
    echo "✅ Quality models downloaded!"
}

# Function to download musicality models  
download_musicality_models() {
    echo "🎼 Downloading Musicality & Style Models (~45MB)..."
    echo "-------------------------------------------------"
    
    echo "📥 Musicality assessment..."
    wget -q --show-progress https://essentia.upf.edu/models/musicality/musicality-discogs-effnet-1.pb -O models/musicality.pb
    
    echo "📥 Decades classification..."
    wget -q --show-progress https://essentia.upf.edu/models/classifiers/decades/decades-discogs-effnet-1.pb -O models/decades.pb
    
    echo "📥 Electronic vs Acoustic..."
    wget -q --show-progress https://essentia.upf.edu/models/classifiers/electronic_acoustic/electronic_acoustic-discogs-effnet-1.pb -O models/electronic_acoustic.pb
    
    echo "📥 Tempo classification..."
    wget -q --show-progress https://essentia.upf.edu/models/tempo/tempo-discogs-effnet-1.pb -O models/tempo_classification.pb
    
    echo "✅ Musicality models downloaded!"
}

# Function to download emotion models
download_emotion_models() {
    echo "⚡ Downloading Advanced Emotion Models (~45MB)..."
    echo "-----------------------------------------------"
    
    echo "📥 Arousal (energy level)..."
    wget -q --show-progress https://essentia.upf.edu/models/arousal/arousal-discogs-effnet-1.pb -O models/arousal.pb
    
    echo "📥 Valence (emotional positivity)..."
    wget -q --show-progress https://essentia.upf.edu/models/valence/valence-discogs-effnet-1.pb -O models/valence.pb
    
    echo "📥 Engagement level..."
    wget -q --show-progress https://essentia.upf.edu/models/engagement/engagement-discogs-effnet-1.pb -O models/engagement.pb
    
    echo "✅ Emotion models downloaded!"
}

# Ensure models directory exists
mkdir -p models

# Interactive selection
read -p "Enter your choice (1-7): " choice

case $choice in
    1)
        download_core_models
        ;;
    2)
        download_instrument_models
        ;;
    3)
        download_quality_models
        ;;
    4)
        download_musicality_models
        ;;
    5)
        download_emotion_models
        ;;
    6)
        echo "📦 Downloading ALL models (~260MB)..."
        download_core_models
        echo ""
        download_instrument_models
        echo ""
        download_quality_models
        echo ""
        download_musicality_models
        echo ""
        download_emotion_models
        ;;
    7)
        echo "🎯 Downloading RECOMMENDED SET (~200MB)..."
        download_core_models
        echo ""
        download_instrument_models
        echo ""
        download_quality_models
        echo ""
        download_musicality_models
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🎉 MODEL DOWNLOAD COMPLETE!"
echo "=========================="
echo ""

# Show what was downloaded
total_models=$(ls -1 models/*.pb 2>/dev/null | wc -l)
total_size=$(du -sh models/ 2>/dev/null | cut -f1)

echo "📊 Downloaded Summary:"
echo "--------------------"
echo "Total models: $total_models"
echo "Total size: $total_size"
echo ""

echo "📁 Downloaded model files:"
ls -la models/*.pb 2>/dev/null | awk '{print "  " $9 " (" $5 " bytes)"}' || echo "  No models found"

echo ""
echo "🚀 Next Steps:"
echo "1. Update essentia_models.py to recognize these models"
echo "2. Run validation script to test ML integration"
echo "3. Start using advanced ML features!"
echo ""
echo "💡 Tip: Models are downloaded but only loaded when you actually use them."
echo "   This saves memory and startup time."
