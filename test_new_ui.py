#!/usr/bin/env python3
"""Test the new UI with a simple file"""

import requests
import json
import sys
import numpy as np
from scipy.io import wavfile
import tempfile
import os

def generate_simple_test_audio():
    """Generate simple test audio for UI testing"""
    sr = 22050
    
    # 10 seconds sound
    t = np.linspace(0, 10, sr * 10)
    sound = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    wavfile.write(temp_file.name, sr, (sound * 32767).astype(np.int16))
    
    return temp_file.name

def test_new_ui():
    """Test the new UI and region cards"""
    
    # Generate test audio
    audio_file = generate_simple_test_audio()
    print(f"📁 Generated test audio: {audio_file}")
    
    try:
        # Upload to API
        url = "http://localhost:8001/api/audio/analyze-enhanced"
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            print("📤 Uploading to API...")
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response received")
            print(f"📊 Success: {result.get('success', False)}")
            
            # Check region analysis for UI
            analysis = result.get('analysis', {})
            region_analysis = analysis.get('region_analysis', {})
            
            if region_analysis and region_analysis.get('regions'):
                print(f"\n🎴 UI REGION CARDS DATA:")
                print(f"   Total regions: {len(region_analysis['regions'])}")
                
                for region in region_analysis['regions']:
                    region_info = region.get('region_info', {})
                    analysis_results = region.get('analysis_results', {})
                    
                    print(f"\n   📊 Region {region_info.get('region_number', 'Unknown')} Card:")
                    print(f"      Type: {region_info.get('content_type', 'Unknown')}")
                    print(f"      Time: {region_info.get('start_time', 0):.1f}s - {region_info.get('end_time', 0):.1f}s")
                    print(f"      Duration: {region_info.get('duration', 0):.1f}s")
                    
                    # Check what analysis results are available for UI
                    if analysis_results.get('key_detection'):
                        print(f"      Key: {analysis_results['key_detection'].get('key', 'Unknown')}")
                    if analysis_results.get('tempo_detection'):
                        print(f"      Tempo: {analysis_results['tempo_detection'].get('tempo', 0):.0f} BPM")
                    if analysis_results.get('downbeat_detection'):
                        print(f"      Downbeats: {analysis_results['downbeat_detection'].get('downbeat_count', 0)}")
                    if analysis_results.get('danceability'):
                        print(f"      Danceability: {analysis_results['danceability'].get('score', 0)*100:.0f}%")
                        
            # Check content analysis for timeline
            content_analysis = analysis.get('content_analysis', {})
            if content_analysis:
                print(f"\n🎯 TIMELINE REGIONS:")
                print(f"   Total regions: {len(content_analysis.get('regions', []))}")
                for i, region in enumerate(content_analysis.get('regions', [])):
                    print(f"   Region {i+1}: {region.get('start', 0):.1f}s-{region.get('end', 0):.1f}s | {region.get('type', '').upper()}")
                    
            print(f"\n🎨 New UI should show:")
            print(f"   - Compact file info bar")
            print(f"   - Clean timeline with region markers only")
            print(f"   - {len(region_analysis.get('regions', []))} region cards in grid")
            print(f"   - No downbeat/chord clutter")
                
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
    test_new_ui()