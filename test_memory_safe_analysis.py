# test_memory_safe_analysis.py
import sys
import os
sys.path.append('/opt/2w12-backend')

from core.audio_analyzer import AudioAnalyzer
import time

def test_memory_safe_method():
    """Test the new memory-safe analysis method with your specific files"""
    
    analyzer = AudioAnalyzer()
    
    # Check if new method exists
    if not hasattr(analyzer, 'comprehensive_analysis_memory_safe'):
        print("❌ New method not found. Please check if it was added correctly.")
        return False
    
    print("✅ New memory-safe method found")
    
    # Test with your specific files
    test_files = [
        ("dnbtest.wav", "Smallest file"),
        ("test48short.wav", "Short version"), 
        ("test48.wav", "Large file that was crashing")
    ]
    
    for file_path, description in test_files:
        if os.path.exists(file_path):
            print(f"\n🧪 Testing {description}: {file_path}")
            
            # Get file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   File size: {file_size_mb:.1f}MB")
            
            # Check memory before
            if hasattr(analyzer, 'get_memory_usage'):
                memory_before = analyzer.get_memory_usage()
                print(f"   Memory before: {memory_before['rss_mb']:.1f}MB RSS, {memory_before['available_system_mb']:.1f}MB available")
            
            try:
                start_time = time.time()
                
                # Test new method
                result = analyzer.comprehensive_analysis_memory_safe(file_path)
                
                processing_time = time.time() - start_time
                
                # Check memory after
                if hasattr(analyzer, 'get_memory_usage'):
                    memory_after = analyzer.get_memory_usage()
                    print(f"   Memory after: {memory_after['rss_mb']:.1f}MB RSS, {memory_after['available_system_mb']:.1f}MB available")
                
                print(f"   Processing time: {processing_time:.2f} seconds")
                
                # Check if we got valid results
                if "analysis" in result and "audio_features" in result:
                    analysis = result["analysis"]
                    audio_features = result["audio_features"]
                    
                    print(f"   ✅ Success!")
                    print(f"      Key: {analysis.get('key', 'N/A')}")
                    print(f"      Tempo: {analysis.get('tempo', 'N/A'):.1f} BPM")
                    print(f"      Duration: {analysis.get('duration', audio_features.get('duration', 'N/A')):.1f}s")
                    print(f"      Processing mode: {audio_features.get('processing_mode', 'unknown')}")
                    
                    # Show additional info for chunked processing
                    if "timeline_analysis" in result:
                        timeline = result["timeline_analysis"]
                        print(f"      Key changes detected: {len(timeline.get('key_changes', []))}")
                        print(f"      Tempo changes detected: {len(timeline.get('tempo_changes', []))}")
                    
                else:
                    print("   ⚠️  Got response but missing expected structure")
                    print(f"      Keys in result: {list(result.keys())}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                import traceback
                traceback.print_exc()
                
        else:
            print(f"⏭️  Skipping {description} - file not found: {file_path}")
    
    return True

def test_comparison_with_existing():
    """Compare new method with existing method on smallest file"""
    
    analyzer = AudioAnalyzer()
    
    # Use dnbtest as the smallest file for comparison
    small_file = "dnbtest.wav"
    
    if not os.path.exists(small_file):
        print("⏭️  No dnbtest.wav found for comparison")
        return
    
    file_size_mb = os.path.getsize(small_file) / (1024 * 1024)
    print(f"\n🔄 Comparing methods on smallest file: {small_file} ({file_size_mb:.1f}MB)")
    
    try:
        # Test existing method
        print("   Testing existing comprehensive_analysis_with_features()...")
        start_time = time.time()
        existing_result = analyzer.comprehensive_analysis_with_features(small_file)
        existing_time = time.time() - start_time
        
        # Test new method  
        print("   Testing new comprehensive_analysis_memory_safe()...")
        start_time = time.time()
        new_result = analyzer.comprehensive_analysis_memory_safe(small_file)
        new_time = time.time() - start_time
        
        print(f"\n📊 Comparison Results:")
        print(f"   Existing method: {existing_time:.2f}s")
        print(f"   New method: {new_time:.2f}s")
        
        # Compare analysis results
        existing_analysis = existing_result.get("analysis", {})
        new_analysis = new_result.get("analysis", {})
        
        print(f"\n   Analysis comparison:")
        print(f"   Key - Existing: {existing_analysis.get('key')}, New: {new_analysis.get('key')}")
        
        existing_tempo = existing_analysis.get('tempo', 0)
        new_tempo = new_analysis.get('tempo', 0)
        print(f"   Tempo - Existing: {existing_tempo:.1f}, New: {new_tempo:.1f}")
        
        # Check if they're reasonably close
        tempo_diff = abs(existing_tempo - new_tempo)
        if tempo_diff < 5:  # Within 5 BPM
            print("   ✅ Tempo analysis consistent between methods")
        else:
            print(f"   ⚠️  Tempo differs by {tempo_diff:.1f} BPM")
        
        if existing_analysis.get('key') == new_analysis.get('key'):
            print("   ✅ Key analysis consistent between methods")
        else:
            print("   ⚠️  Key analysis differs between methods")
            
        # Check response sizes
        import json
        existing_size = len(json.dumps(existing_result))
        new_size = len(json.dumps(new_result))
        print(f"\n   Response sizes:")
        print(f"   Existing: {existing_size:,} characters")
        print(f"   New: {new_size:,} characters")
        print(f"   Size reduction: {((existing_size - new_size) / existing_size * 100):.1f}%")
            
    except Exception as e:
        print(f"   ❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Memory-Safe Audio Analysis Method")
    print("=" * 60)
    
    if test_memory_safe_method():
        test_comparison_with_existing()
    
    print("\n✅ Testing complete!")
    print("\nNext: If tests pass, we'll integrate this into your visualization endpoint.")
