#!/usr/bin/env python3
# Test script for content-aware analysis
import requests
import json
import time

def test_content_aware_analysis():
    """Test the new content-aware analysis pipeline"""
    
    print("🎯 Testing Content-Aware Analysis Implementation")
    print("=" * 60)
    
    # Generate a simple test audio file with silence + music + silence
    import numpy as np
    import scipy.io.wavfile
    import tempfile
    import os
    
    # Create test audio: 2s silence + 10s music + 2s silence = 14s total
    sample_rate = 22050
    
    # 2 seconds silence
    silence1 = np.zeros(int(2.0 * sample_rate))
    
    # 10 seconds of test music (sine waves for chords)
    duration_music = 10.0
    t = np.linspace(0, duration_music, int(duration_music * sample_rate))
    
    # Create a simple chord progression (C major chord)
    freq_c = 261.63  # C4
    freq_e = 329.63  # E4  
    freq_g = 392.00  # G4
    
    music = (
        0.3 * np.sin(2 * np.pi * freq_c * t) +  # C
        0.3 * np.sin(2 * np.pi * freq_e * t) +  # E
        0.3 * np.sin(2 * np.pi * freq_g * t)    # G
    )
    
    # 2 seconds silence
    silence2 = np.zeros(int(2.0 * sample_rate))
    
    # Combine all parts
    full_audio = np.concatenate([silence1, music, silence2])
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
        scipy.io.wavfile.write(tmp_file_path, sample_rate, (full_audio * 32767).astype(np.int16))
    
    print(f"📊 Test audio created: 14s total (2s silence + 10s music + 2s silence)")
    print(f"💾 Temporary file: {tmp_file_path}")
    
    try:
        # Upload and analyze
        print("\n🚀 Starting content-aware analysis...")
        start_time = time.time()
        
        with open(tmp_file_path, 'rb') as audio_file:
            files = {'file': ('test_audio.wav', audio_file, 'audio/wav')}
            response = requests.post(
                'http://localhost:8001/api/audio/analyze-enhanced',
                files=files,
                timeout=60
            )
        
        analysis_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Analysis completed in {analysis_time:.2f}s")
            
            # Check content analysis results (nested under 'analysis' key)
            if 'analysis' in result and 'content_analysis' in result['analysis']:
                content_analysis = result['analysis']['content_analysis']
                print(f"\n🎯 CONTENT-AWARE ANALYSIS RESULTS:")
                print(f"   📊 Total regions detected: {len(content_analysis['regions'])}")
                print(f"   🎵 Musical regions: {content_analysis['musical_regions_count']}")
                print(f"   ⏱️ Musical duration: {content_analysis['total_musical_duration']:.1f}s")
                print(f"   ⚡ Efficiency: {content_analysis['efficiency_stats']['efficiency_percentage']:.1f}% analyzed")
                print(f"   💾 Time saved: {content_analysis['efficiency_stats']['time_saved_percentage']:.1f}%")
                
                print(f"\n📋 DETECTED REGIONS:")
                for i, region in enumerate(content_analysis['regions']):
                    status = "✅ ANALYZED" if region['analyzed'] else "⏭️ SKIPPED"
                    print(f"   Region {i+1}: {region['start']:.1f}s-{region['end']:.1f}s | {region['type']} | {status}")
                
                # Check if analysis was actually content-aware
                total_duration = 14.0
                musical_duration = content_analysis['total_musical_duration']
                expected_musical = 10.0  # We created 10s of music
                
                if abs(musical_duration - expected_musical) < 1.0:
                    print(f"\n✅ CONTENT DETECTION SUCCESS!")
                    print(f"   Expected ~10s music, detected {musical_duration:.1f}s")
                else:
                    print(f"\n⚠️ CONTENT DETECTION NEEDS TUNING")
                    print(f"   Expected ~10s music, detected {musical_duration:.1f}s")
                
            else:
                print(f"❌ No content_analysis found in result")
                
            # Check architecture
            analysis = result.get('analysis', {})
            if 'architecture' in analysis:
                print(f"\n🏗️ Architecture: {analysis['architecture']}")
                
            # Check analysis pipeline
            if 'analysis_pipeline' in analysis:
                print(f"⚙️ Pipeline: {analysis['analysis_pipeline']}")
            
            print(f"\n🎵 OTHER ANALYSIS RESULTS:")
            if 'ml_key' in analysis:
                print(f"   🎹 Key detected: {analysis.get('ml_key', 'N/A')}")
            if 'madmom_downbeat_count' in analysis:
                print(f"   🥁 Downbeats: {analysis.get('madmom_downbeat_count', 'N/A')}")
            if 'content_aware_filtering' in analysis:
                filtering = analysis['content_aware_filtering']
                print(f"   🎯 Downbeat filtering: {filtering['original_downbeats']} → {filtering['filtered_downbeats']}")
                
        else:
            print(f"❌ Analysis failed: HTTP {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
            print(f"\n🧹 Cleaned up temporary file")

if __name__ == "__main__":
    test_content_aware_analysis()