# LIBHUNT.md - Audio Processing Libraries & Models Research

**Date**: July 16, 2025  
**Session**: Library Research & Modern Audio Tools Investigation  
**Status**: Research Complete - Ready for Implementation Discussion

---

## 🎯 **RESEARCH OBJECTIVE**

Investigation of latest open source audio processing libraries and models from major tech companies (Google, Meta, Microsoft, Spotify) to enhance the 2W12 Audio Analysis Platform with:
- Faster pitch detection alternatives
- Real-time audio playback capabilities  
- Polyphonic MIDI extraction
- Advanced chord/key detection from MIDI data

---

## 🔬 **CURRENT 2W12 PERFORMANCE BASELINE**

### **Processing Pipeline Analysis (70.8s audio file):**
```
Total Processing: 18s (1.25x realtime)
├── AudioFlux: 0.028s (0.2%)
├── Essentia ML: 0.96s (5.3%)  
├── Madmom: 7.4s (41.1%) ← BOTTLENECK
├── Visualization: 0.2s (1.1%)
├── Chords: 0.5s (2.8%)
└── Other/Overhead: 8.9s (49.5%)
```

### **Current Tech Stack:**
- **Pitch Detection**: CREPE (50MB, 95%+ accuracy, 0.5s for 10s audio)
- **Tempo Analysis**: Madmom + Essentia RhythmExtractor2013
- **Chord Detection**: Template matching approach
- **Visualization**: Canvas-based with performance issues

---

## 🏆 **DISCOVERED LIBRARIES & MODELS**

### **1. BASIC PITCH (Spotify) - Audio-to-MIDI Conversion**

**Overview**: Spotify's lightweight, lightning-fast audio-to-MIDI converter with polyphonic capabilities

**Key Features:**
- **Polyphonic Detection**: Multiple simultaneous notes (vs monophonic)
- **Pitch Bend Support**: Vibrato, glissando, slides, bends
- **Instrument Agnostic**: Piano, guitar, voice, any instrument
- **Browser Compatible**: Runs in web browser
- **Lightweight**: <20MB memory, <17K parameters

**Performance:**
- **Speed**: Faster than realtime processing
- **Size**: <20MB (vs CREPE's 50MB)
- **Accuracy**: ~85% note detection (vs 95%+ for monophonic)

**Use Cases for 2W12:**
- Extract MIDI notes from audio files
- Generate polyphonic chord progressions
- Create exportable MIDI files
- Enable DAW integration

**Integration Example:**
```python
from basic_pitch.inference import predict

# Convert audio to MIDI
model_output, midi_data, note_events = predict(audio_file)

# Results: Complete MIDI representation
# - Note events with start_time, end_time, pitch, velocity
# - Pitch bend information
# - Polyphonic chord detection
```

### **2. SPICE (Google Magenta) - Pitch Recognition**

**Overview**: Google's Self-Supervised Pitch Estimation model for lightweight pitch detection

**Key Features:**
- **Self-supervised Learning**: Trained without manual labeling
- **TensorFlow Hub Integration**: Easy model loading
- **Real-time Capable**: Optimized for speed
- **Monophonic Focus**: Single pitch line extraction

**Performance:**
- **Speed**: ~0.1s for 10s audio (5x faster than CREPE)
- **Size**: ~10MB model
- **Accuracy**: ~85% pitch detection
- **Input**: 16kHz mono audio

**Use Cases for 2W12:**
- Replace CREPE for basic pitch detection
- Real-time pitch tracking
- Lightweight key detection alternative

**Integration Example:**
```python
import tensorflow_hub as hub

# Load SPICE model
spice_model = hub.load("https://tfhub.dev/google/spice/2")

# Fast pitch detection
pitch_predictions = spice_model(audio_16khz)
```

### **3. PEDALBOARD (Spotify) - Audio Effects & Playback**

**Overview**: Spotify's C++ audio effects library with Python bindings for real-time processing

**Key Features:**
- **300x faster** than pySoX for audio transforms
- **4x faster** file reading than librosa
- **VST3/Audio Unit** plugin support
- **Real-time processing** capability
- **Memory efficient** streaming

**Performance Comparison:**
```python
# Traditional approach (SLOW)
import sox
tfm = sox.Transformer()
tfm.reverb(50)  # ~300ms for 10s audio

# Pedalboard approach (FAST)  
import pedalboard
board = pedalboard.Pedalboard([pedalboard.Reverb(room_size=0.5)])
output = board(audio, sample_rate)  # ~1ms for same audio
```

**Use Cases for 2W12:**
- **Real-time audio playback** without effects overhead
- **Data augmentation** for ML training (300x faster)
- **Interactive audio** manipulation
- **Streaming audio** processing

**Current Usage at Spotify:**
- Powers Spotify's AI DJ
- AI Voice Translation processing
- ML model data augmentation

### **4. AUDIOCRAFT SUITE (Meta) - Generative Audio**

**Overview**: Meta's comprehensive audio generation framework

**Components:**
- **MusicGen**: Text-to-music generation
- **AudioGen**: Text-to-sound generation  
- **EnCodec**: Neural audio codec
- **AudioSeal**: Audio watermarking
- **JASCO**: Chord/melody-conditioned generation

**JASCO Deep Dive:**
```python
# JASCO generates music from:
jasco_model.generate(
    chords=["C", "Am", "F", "G"],           # Chord progression
    melody=[60, 64, 67, 72],               # MIDI note sequence
    drums="basic_rock_pattern",            # Rhythm pattern
    text="upbeat jazz style"               # Style description
)
```

**2W12 Integration Potential:**
- **Reverse Engineering**: Use detected chords to generate variations
- **Music Completion**: Create full tracks from analysis results
- **Style Transfer**: Convert patterns to different genres

### **5. MAGENTA REALTIME (Google) - Live Music Generation**

**Overview**: Google's 800M parameter live music model for real-time generation

**Key Features:**
- **Real-time Generation**: Continuous music streams
- **Style Manipulation**: Text/audio prompt conditioning
- **Local Execution**: Runs on consumer hardware
- **Block Autoregression**: Sequential chunk generation

**Use Cases for 2W12:**
- **Live accompaniment** based on detected chords
- **Real-time harmony** generation
- **Interactive music** creation

### **6. ECLIPSA AUDIO (Google + Samsung) - Spatial Audio**

**Overview**: Open-source spatial audio format with YouTube integration

**Key Features:**
- **Spatial Audio**: 3D sound positioning
- **YouTube Integration**: Native support in 2025
- **Pro Tools Plugin**: Free AVID integration
- **Samsung TV**: Native playback support

**2W12 Integration Potential:**
- **Spatial analysis**: 3D audio visualization
- **Export formats**: Enhanced audio output

---

## 🎼 **MIDI-TO-CHORD/KEY DETECTION RESEARCH**

### **Problem**: Basic Pitch outputs raw MIDI notes, not chord names or keys

### **Solution**: Music Theory Libraries

#### **1. MUSIC21 (MIT) - Comprehensive Music Analysis**

**Key Detection Algorithm:**
```python
import music21

# Load MIDI from Basic Pitch
score = music21.converter.parse(midi_data)

# Krumhansl-Schmuckler key detection
key = score.analyze('key')
print(f"Key: {key.tonic.name} {key.mode}")  # "C major"

# Chord analysis
chords = score.chordify()
for chord in chords.recurse().getElementsByClass('Chord'):
    print(f"Time: {chord.offset}, Chord: {chord.pitchedCommonName}")
```

**Features:**
- **Krumhansl-Schmuckler Algorithm**: Industry standard for key detection
- **Automatic Chord Recognition**: From polyphonic MIDI
- **Music Theory Integration**: Comprehensive analysis tools
- **Academic Backing**: MIT-developed, peer-reviewed

#### **2. CUSTOM CHORD DETECTION ALGORITHM**

**Lightweight Alternative:**
```python
def detect_chord_from_notes(notes):
    """Detect chord type from pitch classes"""
    intervals = sorted(set(notes))
    
    # Major triad: 0, 4, 7 (C-E-G)
    if intervals == [0, 4, 7]:
        return "major"
    # Minor triad: 0, 3, 7 (C-Eb-G)  
    elif intervals == [0, 3, 7]:
        return "minor"
    # Dominant 7th: 0, 4, 7, 10
    elif intervals == [0, 4, 7, 10]:
        return "dominant7"
    # Add more chord types...
    
    return "complex"
```

#### **3. KRUMHANSL-SCHMUCKLER KEY DETECTION**

**Algorithm Implementation:**
```python
def detect_key_from_midi(midi_notes):
    """Detect key using pitch class histogram"""
    
    # Krumhansl-Schmuckler weights
    major_weights = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 
                    2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_weights = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 
                    2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    # Create pitch class histogram
    pitch_histogram = [0] * 12
    for note in midi_notes:
        pitch_histogram[note.pitch % 12] += note.duration
    
    # Test all 24 keys (12 major + 12 minor)
    correlations = []
    for tonic in range(12):
        major_corr = correlate(pitch_histogram, rotate(major_weights, tonic))
        minor_corr = correlate(pitch_histogram, rotate(minor_weights, tonic))
        correlations.extend([(major_corr, tonic, 'major'), 
                           (minor_corr, tonic, 'minor')])
    
    # Best correlation
    best_corr, best_tonic, best_mode = max(correlations, key=lambda x: x[0])
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{note_names[best_tonic]} {best_mode}"
```

---

## 🚀 **INTEGRATION ROADMAP**

### **Phase 1: Basic Pitch Integration**
- Add polyphonic MIDI extraction
- Implement Music21 chord detection
- Create MIDI export functionality

### **Phase 2: Performance Optimization**
- Replace CREPE with SPICE for basic pitch detection
- Implement Pedalboard for real-time playback
- Optimize canvas rendering for visualization

### **Phase 3: Advanced Features**
- JASCO integration for music generation
- Spatial audio analysis with Eclipsa
- Real-time music interaction

---

## 📊 **PERFORMANCE COMPARISON**

| Tool | Speed | Memory | Accuracy | Capabilities |
|------|-------|---------|----------|-------------|
| **Current 2W12** | 1.25x realtime | ~200MB | 95%+ | Full pipeline |
| **Basic Pitch** | <1.0x realtime | <20MB | 85%+ | Polyphonic MIDI |
| **SPICE** | Real-time | ~10MB | 85%+ | Monophonic pitch |
| **CREPE** | 0.5s/10s | 50MB | 95%+ | High-precision pitch |
| **Pedalboard** | 300x faster | Minimal | N/A | Audio processing |

---

## 🎯 **RECOMMENDATIONS**

### **Immediate Actions:**
1. **Fix UI lag**: Optimize canvas rendering (batch drawing)
2. **Pedalboard integration**: Real-time playback capability
3. **Basic Pitch trial**: Test polyphonic MIDI extraction

### **Medium-term Goals:**
1. **Hybrid approach**: Combine CREPE accuracy with Basic Pitch polyphony
2. **Music21 integration**: Comprehensive chord/key analysis
3. **Performance optimization**: Selective tool usage based on file characteristics

### **Long-term Vision:**
1. **Complete metadata platform**: Audio analysis + MIDI + generation
2. **Real-time capabilities**: Live music interaction
3. **DAW integration**: Professional workflow compatibility

---

## 🔍 **IDENTIFIED ISSUES**

### **UI Performance Problem:**
- **Root Cause**: Canvas rendering bottleneck in JavaScript
- **Location**: `drawWaveform()` function - individual stroke operations
- **Impact**: Lag between console results and visualization
- **Solution**: Batch drawing operations, optimize text rendering

### **Current Bottlenecks:**
1. **Madmom**: 7.4s for 66 downbeats (41% of processing time)
2. **Canvas rendering**: Individual stroke operations for each downbeat
3. **Text rendering**: Beat number labels cause additional delays

---

## 💡 **NEXT SESSION PRIORITIES**

1. **Library Discussion**: Evaluate integration priorities
2. **Performance Fixes**: Address UI rendering lag
3. **Basic Pitch Trial**: Test polyphonic MIDI extraction
4. **Pedalboard Integration**: Real-time playback implementation
5. **Hybrid Architecture**: Combine best tools for optimal results

---

## 📚 **RESEARCH SOURCES**

- Basic Pitch: https://github.com/spotify/basic-pitch
- SPICE: https://tfhub.dev/google/spice/2
- Pedalboard: https://github.com/spotify/pedalboard
- Music21: https://web.mit.edu/music21/
- AudioCraft: https://github.com/facebookresearch/audiocraft
- Magenta: https://magenta.tensorflow.org/

---

**End of Research Document**  
*Research completed: July 16, 2025*  
*Next session: Implementation discussion and priority setting*  
*Status: Ready for development decisions*