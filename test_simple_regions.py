#!/usr/bin/env python3
"""Test simple region detection system"""

import requests
import json
import sys
import numpy as np
from scipy.io import wavfile
import tempfile
import os

def generate_test_audio():
    """Generate simple test audio with silence + sound + silence pattern"""
    sr = 22050
    
    # 2 seconds silence
    silence1 = np.zeros(sr * 2)
    
    # 5 seconds sine wave (sound)
    t = np.linspace(0, 5, sr * 5)
    sound = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # 2 seconds silence
    silence2 = np.zeros(sr * 2)
    
    # Combine
    audio = np.concatenate([silence1, sound, silence2])
    
    # Save to temp file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    wavfile.write(temp_file.name, sr, (audio * 32767).astype(np.int16))
    
    return temp_file.name

def test_simple_region_detection():
    """Test the simplified region detection API"""
    
    # Generate test audio
    audio_file = generate_test_audio()
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
                
                # Expected: 3 regions (silence, sound, silence) with only middle region analyzed
                regions = content_analysis.get('regions', [])
                if len(regions) == 3:
                    print(f"\n✅ Expected 3 regions found")
                    if regions[0]['type'] == 'silence' and regions[1]['type'] == 'sound' and regions[2]['type'] == 'silence':
                        print(f"✅ Correct sequence: silence -> sound -> silence")
                    if not regions[0]['analyzed'] and regions[1]['analyzed'] and not regions[2]['analyzed']:
                        print(f"✅ Only sound region will be analyzed")
                else:
                    print(f"❌ Expected 3 regions, got {len(regions)}")
                    
            else:
                print("❌ No content analysis found in response")
                
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
    test_simple_region_detection()