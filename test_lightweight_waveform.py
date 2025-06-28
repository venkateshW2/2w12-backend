# test_lightweight_waveform.py
import sys
import os
sys.path.append('/opt/2w12-backend')

from core.audio_analyzer import AudioAnalyzer
import json
import time

def test_lightweight_waveform_method():
    """Test the new lightweight waveform method specifically"""
    
    analyzer = AudioAnalyzer()
    
    # Check if new method exists
    if not hasattr(analyzer, 'get_lightweight_waveform'):
        print("❌ get_lightweight_waveform method not found!")
        return False
    
    print("✅ get_lightweight_waveform method found")
    
    test_files = ["dnbtest.wav", "test48short.wav", "test48.wav"]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"\n🧪 Testing lightweight waveform: {file_path} ({file_size_mb:.1f}MB)")
            
            try:
                start_time = time.time()
                
                # Test the new lightweight waveform method directly
                waveform_result = analyzer.get_lightweight_waveform(file_path)
                
                processing_time = time.time() - start_time
                
                print(f"   ⏱️  Processing time: {processing_time:.3f} seconds")
                print(f"   📊 Waveform points: {waveform_result.get('points', 'N/A')}")
                print(f"   ⏰ Duration: {waveform_result.get('duration', 'N/A'):.1f}s")
                print(f"   🎵 Sample rate: {waveform_result.get('sample_rate', 'N/A')}Hz")
                
                # Check JSON size
                json_str = json.dumps(waveform_result)
                json_size_kb = len(json_str) / 1024
                print(f"   💾 JSON size: {json_size_kb:.1f}KB")
                
                # Verify waveform data
                waveform_data = waveform_result.get('waveform_data', [])
                if len(waveform_data) > 0:
                    print(f"   ✅ Waveform data generated: {len(waveform_data)} points")
                    print(f"   📈 Amplitude range: {min(waveform_data):.3f} to {max(waveform_data):.3f}")
                else:
                    print(f"   ❌ No waveform data generated")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"⏭️  File not found: {file_path}")
    
    return True

def test_memory_safe_with_new_waveform():
    """Test the updated comprehensive_analysis_memory_safe method"""
    
    analyzer = AudioAnalyzer()
    
    print(f"\n🔄 Testing updated memory-safe analysis...")
    
    test_files = ["dnbtest.wav", "test48short.wav", "test48.wav"]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"\n🧪 Testing updated method: {file_path} ({file_size_mb:.1f}MB)")
            
            try:
                start_time = time.time()
                
                # Test updated comprehensive_analysis_memory_safe
                result = analyzer.comprehensive_analysis_memory_safe(file_path)
                
                processing_time = time.time() - start_time
                print(f"   ⏱️  Total processing time: {processing_time:.2f} seconds")
                
                # Check if visualization data exists
                if "visualization" in result:
                    viz = result["visualization"]
                    print(f"   ✅ Visualization data present")
                    print(f"   📊 Waveform points: {viz.get('points', 'N/A')}")
                    print(f"   ⏰ Duration: {viz.get('duration', 'N/A'):.1f}s")
                    
                    # Check size of visualization data
                    viz_json = json.dumps(viz)
                    viz_size_kb = len(viz_json) / 1024
                    print(f"   💾 Visualization JSON size: {viz_size_kb:.1f}KB")
                    
                    # Check total response size
                    total_json = json.dumps(result)
                    total_size_kb = len(total_json) / 1024
                    print(f"   📦 Total response size: {total_size_kb:.1f}KB")
                    
                else:
                    print(f"   ⚠️  No visualization data in response")
                    print(f"   🔍 Available keys: {list(result.keys())}")
                
                # Check analysis results
                if "analysis" in result:
                    analysis = result["analysis"]
                    print(f"   🎵 Analysis: Key={analysis.get('key', 'N/A')}, Tempo={analysis.get('tempo', 'N/A'):.1f}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Lightweight Waveform Implementation")
    print("=" * 60)
    
    # Test 1: Direct waveform method
    if test_lightweight_waveform_method():
        print("\n" + "="*60)
        # Test 2: Updated memory-safe method
        test_memory_safe_with_new_waveform()
    
    print("\n✅ Testing complete!")