#!/usr/bin/env python3
"""Debug the actual response structure"""

import requests
import json
import sys
import numpy as np
from scipy.io import wavfile
import tempfile
import os

def generate_long_test_audio():
    """Generate longer test audio to see multiple regions"""
    sr = 22050
    
    # 10 seconds sound
    t1 = np.linspace(0, 10, sr * 10)
    sound1 = 0.5 * np.sin(2 * np.pi * 440 * t1)
    
    # 30 seconds silence (exceeds 25s threshold)
    silence = np.zeros(sr * 30)
    
    # 10 seconds sound
    t2 = np.linspace(0, 10, sr * 10)
    sound2 = 0.5 * np.sin(2 * np.pi * 880 * t2)
    
    # Another 30 seconds silence
    silence2 = np.zeros(sr * 30)
    
    # Final 10 seconds sound
    t3 = np.linspace(0, 10, sr * 10)
    sound3 = 0.5 * np.sin(2 * np.pi * 1320 * t3)
    
    # Combine
    audio = np.concatenate([sound1, silence, sound2, silence2, sound3])
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    wavfile.write(temp_file.name, sr, (audio * 32767).astype(np.int16))
    
    return temp_file.name

def debug_response_structure():
    """Debug the complete response structure"""
    
    # Generate test audio
    audio_file = generate_long_test_audio()
    print(f"📁 Generated test audio: {audio_file}")
    
    try:
        # Upload to API
        url = "http://localhost:8001/api/audio/analyze-enhanced"
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            print("📤 Uploading to API...")
            response = requests.post(url, files=files, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response received")
            print(f"📊 Success: {result.get('success', False)}")
            
            # Save full response for inspection
            with open('/tmp/full_response.json', 'w') as f:
                json.dump(result, f, indent=2)
            print(f"💾 Full response saved to /tmp/full_response.json")
            
            # Check main structure
            print(f"\n📋 MAIN RESPONSE STRUCTURE:")
            for key in result.keys():
                if isinstance(result[key], dict):
                    print(f"   {key}: {len(result[key])} items")
                elif isinstance(result[key], list):
                    print(f"   {key}: {len(result[key])} items")
                else:
                    print(f"   {key}: {result[key]}")
            
            # Check analysis structure
            analysis = result.get('analysis', {})
            print(f"\n📊 ANALYSIS STRUCTURE:")
            for key in analysis.keys():
                if isinstance(analysis[key], dict):
                    print(f"   analysis.{key}: {len(analysis[key])} items")
                elif isinstance(analysis[key], list):
                    print(f"   analysis.{key}: {len(analysis[key])} items")
                else:
                    print(f"   analysis.{key}: {analysis[key]}")
            
            # Check content analysis
            content_analysis = analysis.get('content_analysis', {})
            if content_analysis:
                print(f"\n🔊 CONTENT ANALYSIS:")
                print(f"   Regions: {len(content_analysis.get('regions', []))}")
                print(f"   Sound regions count: {content_analysis.get('sound_regions_count', 0)}")
                print(f"   Total sound duration: {content_analysis.get('total_sound_duration', 0):.1f}s")
                
                regions = content_analysis.get('regions', [])
                for i, region in enumerate(regions):
                    print(f"   Region {i+1}: {region.get('start', 0):.1f}s-{region.get('end', 0):.1f}s | {region.get('type', '')} | analyzed: {region.get('analyzed', False)}")
            
            # Check visualization structure
            visualization = result.get('visualization', {})
            print(f"\n🎨 VISUALIZATION STRUCTURE:")
            for key in visualization.keys():
                if isinstance(visualization[key], dict):
                    print(f"   visualization.{key}: {len(visualization[key])} items")
                elif isinstance(visualization[key], list):
                    print(f"   visualization.{key}: {len(visualization[key])} items")
                else:
                    print(f"   visualization.{key}: {visualization[key]}")
            
            # Check chords specifically
            chords = visualization.get('chords', {})
            if chords:
                print(f"\n🎵 CHORD ANALYSIS:")
                chord_events = chords.get('events', [])
                print(f"   Chord events: {len(chord_events)}")
                if chord_events:
                    print(f"   First chord: {chord_events[0]}")
                    print(f"   Last chord: {chord_events[-1]}")
                    
                    # Check chord timeline
                    total_duration = visualization.get('waveform', {}).get('duration', 0)
                    chord_coverage = sum(event.get('end', 0) - event.get('start', 0) for event in chord_events)
                    print(f"   Chord timeline coverage: {chord_coverage:.1f}s / {total_duration:.1f}s")
                    
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    finally:
        # Clean up
        if os.path.exists(audio_file):
            os.unlink(audio_file)

if __name__ == "__main__":
    debug_response_structure()