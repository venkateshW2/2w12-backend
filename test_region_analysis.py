#!/usr/bin/env python3
"""Test region analysis results"""

import requests
import json
import sys
import numpy as np
from scipy.io import wavfile
import tempfile
import os

def generate_test_audio_with_long_silence():
    """Generate test audio with long silence to test 25-second threshold"""
    sr = 22050
    
    # 5 seconds sound
    t1 = np.linspace(0, 5, sr * 5)
    sound1 = 0.5 * np.sin(2 * np.pi * 440 * t1)
    
    # 30 seconds silence (exceeds 25s threshold)
    silence = np.zeros(sr * 30)
    
    # 5 seconds sound
    t2 = np.linspace(0, 5, sr * 5)
    sound2 = 0.5 * np.sin(2 * np.pi * 880 * t2)
    
    # Combine
    audio = np.concatenate([sound1, silence, sound2])
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    wavfile.write(temp_file.name, sr, (audio * 32767).astype(np.int16))
    
    return temp_file.name

def test_region_analysis():
    """Test the region analysis system"""
    
    # Generate test audio
    audio_file = generate_test_audio_with_long_silence()
    print(f"📁 Generated test audio: {audio_file}")
    
    try:
        # Upload to API
        url = "http://localhost:8001/api/audio/analyze-enhanced"
        
        with open(audio_file, 'rb') as f:
            files = {'file': f}
            print("📤 Uploading to API...")
            response = requests.post(url, files=files, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response received")
            print(f"📊 Success: {result.get('success', False)}")
            
            # Check content analysis
            analysis = result.get('analysis', {})
            content_analysis = analysis.get('content_analysis', {})
            
            if content_analysis:
                print(f"\n🔊 REGION ANALYSIS:")
                print(f"   Regions found: {len(content_analysis.get('regions', []))}")
                print(f"   Sound regions: {content_analysis.get('sound_regions_count', 0)}")
                print(f"   Sound duration: {content_analysis.get('total_sound_duration', 0):.1f}s")
                print(f"   Coverage: {content_analysis.get('coverage_stats', {}).get('coverage_percentage', 0):.1f}%")
                
                print(f"\n📋 REGIONS DETECTED:")
                for i, region in enumerate(content_analysis.get('regions', [])):
                    status = "🔊 ANALYZE" if region.get('analyzed') else "🔇 SKIP"
                    print(f"   Region {i+1}: {region.get('start', 0):.1f}s - {region.get('end', 0):.1f}s | {region.get('type', '').upper()} | {status}")
                
            # Check region analysis results
            region_analysis = analysis.get('region_analysis', {})
            if region_analysis and region_analysis.get('regions'):
                print(f"\n🔍 INDIVIDUAL REGION ANALYSIS:")
                print(f"   Total regions analyzed: {len(region_analysis['regions'])}")
                print(f"   Total sound duration: {region_analysis.get('total_sound_duration', 0):.1f}s")
                
                for region in region_analysis['regions']:
                    region_info = region.get('region_info', {})
                    analysis_results = region.get('analysis_results', {})
                    
                    print(f"\n   📊 Region {region_info.get('region_number', 'Unknown')}:")
                    print(f"      Duration: {region_info.get('duration', 0):.1f}s")
                    print(f"      Type: {region_info.get('content_type', 'Unknown')}")
                    print(f"      Confidence: {region_info.get('confidence', 0):.2f}")
                    
                    if analysis_results.get('key_detection'):
                        key_info = analysis_results['key_detection']
                        print(f"      Key: {key_info.get('key', 'Unknown')} (confidence: {key_info.get('confidence', 0):.2f})")
                    
                    if analysis_results.get('tempo_detection'):
                        tempo_info = analysis_results['tempo_detection']
                        print(f"      Tempo: {tempo_info.get('tempo', 0):.1f} BPM (confidence: {tempo_info.get('confidence', 0):.2f})")
                        
                    if analysis_results.get('downbeat_detection'):
                        downbeat_info = analysis_results['downbeat_detection']
                        print(f"      Downbeats: {downbeat_info.get('downbeat_count', 0)} detected")
                        
                    if analysis_results.get('danceability'):
                        dance_info = analysis_results['danceability']
                        print(f"      Danceability: {dance_info.get('score', 0):.2f} (confidence: {dance_info.get('confidence', 0):.2f})")
                        
            else:
                print("❌ No region analysis results found")
                
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
    test_region_analysis()