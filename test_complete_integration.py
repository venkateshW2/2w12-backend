"""
Complete integration test for new visualization and streaming features
"""

import asyncio
import aiohttp
import json
import time
import sys
import os

async def test_complete_workflow():
    base_url = "http://localhost:8001"
    
    print("🧪 Starting complete integration test...")
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    print("✅ Health check passed")
                else:
                    print(f"❌ Health check failed with status {response.status}")
                    return
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("💡 Make sure the server is running with: python main.py")
            return
    
    # Test 2: File upload and analysis
    print("\n2. Testing file upload and analysis...")
    
    # Find available test audio files
    test_files = [
        "test48.wav",
        "test.wav", 
        "dnbtest.wav",
        "large_test_5min.wav"
    ]
    
    test_file_path = None
    for filename in test_files:
        if os.path.exists(filename):
            test_file_path = filename
            break
    
    if not test_file_path:
        print(f"⚠️  No test audio files found. Available files:")
        for f in test_files:
            exists = "✅" if os.path.exists(f) else "❌"
            print(f"   {exists} {f}")
        print("\n💡 Please ensure you have a test audio file to complete integration test")
        return
    
    file_size = os.path.getsize(test_file_path) / (1024*1024)
    print(f"📁 Using test file: {test_file_path} ({file_size:.1f}MB)")
    
    async with aiohttp.ClientSession() as session:
        try:
            with open(test_file_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=os.path.basename(test_file_path))
                
                print("📤 Uploading and analyzing file...")
                start_time = time.time()
                
                # Try the visualization endpoint first
                try:
                    async with session.post(f"{base_url}/api/visualization/analyze-complete", 
                                          data=data, timeout=300) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            processing_time = time.time() - start_time
                            
                            print(f"✅ Analysis completed in {processing_time:.1f}s")
                            
                            if 'processing_time' in result:
                                print(f"   Server processing time: {result['processing_time']:.1f}s")
                            
                            if 'playback' in result and 'file_id' in result['playback']:
                                file_id = result['playback']['file_id']
                                print(f"   File ID: {file_id}")
                                
                                if 'visualization' in result:
                                    viz = result['visualization']
                                    if 'waveform' in viz:
                                        print(f"   Waveform points: {len(viz['waveform'].get('overview', {}).get('data', []))}")
                                    if 'spectrogram' in viz:
                                        print(f"   Spectrogram shape: {viz['spectrogram'].get('shape', 'N/A')}")
                                
                                # Test streaming if file_id is available
                                await test_streaming(session, base_url, file_id)
                                
                            else:
                                print("⚠️  No file_id in response, skipping streaming test")
                            
                        elif response.status == 404:
                            print("❌ Visualization endpoint not found")
                            print("💡 This suggests Phase 1 backend enhancement is not complete")
                            await test_existing_endpoints(session, base_url, test_file_path)
                            
                        else:
                            print(f"❌ Analysis failed with status {response.status}")
                            error_text = await response.text()
                            print(f"   Error: {error_text}")
                            await test_existing_endpoints(session, base_url, test_file_path)
                            
                except asyncio.TimeoutError:
                    print("❌ Request timed out after 5 minutes")
                except Exception as e:
                    print(f"❌ Request failed: {e}")
                    await test_existing_endpoints(session, base_url, test_file_path)
                    
        except Exception as e:
            print(f"❌ File operation failed: {e}")

async def test_streaming(session, base_url, file_id):
    """Test streaming functionality"""
    print("\n3. Testing file status...")
    try:
        async with session.get(f"{base_url}/api/visualization/status/{file_id}") as status_response:
            if status_response.status == 200:
                status_data = await status_response.json()
                print(f"✅ File status: {status_data.get('status', 'unknown')}")
            else:
                print(f"⚠️  File status check returned {status_response.status}")
    except:
        print("⚠️  File status endpoint not available")
    
    print("\n4. Testing audio streaming...")
    try:
        stream_url = f"{base_url}/api/streaming/audio/{file_id}"
        async with session.get(stream_url) as stream_response:
            if stream_response.status == 200:
                chunk = await stream_response.content.read(1024)
                print(f"✅ Audio streaming works (received {len(chunk)} bytes)")
                
                # Test range requests
                print("\n5. Testing range requests...")
                headers = {"Range": "bytes=0-1023"}
                async with session.get(stream_url, headers=headers) as range_response:
                    if range_response.status == 206:
                        print("✅ Range requests working")
                    else:
                        print(f"⚠️  Range requests returned {range_response.status}")
                        
            else:
                print(f"⚠️  Audio streaming returned {stream_response.status}")
    except:
        print("⚠️  Streaming endpoints not available")

async def test_existing_endpoints(session, base_url, test_file_path):
    """Test existing audio analysis endpoints"""
    print("\n🔄 Testing existing audio analysis endpoints...")
    
    try:
        with open(test_file_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('file', f, filename=os.path.basename(test_file_path))
            
            # Test existing analyze endpoint
            async with session.post(f"{base_url}/analyze", data=data, timeout=300) as response:
                if response.status == 200:
                    result = await response.json()
                    print("✅ Existing /analyze endpoint works")
                    print(f"   Key: {result.get('key', 'N/A')}")
                    print(f"   Tempo: {result.get('tempo', 'N/A')}")
                else:
                    print(f"⚠️  /analyze endpoint returned {response.status}")
                    
    except Exception as e:
        print(f"⚠️  Error testing existing endpoints: {e}")

if __name__ == "__main__":
    print("🚀 2W12 Backend Integration Test")
    print("=" * 50)
    asyncio.run(test_complete_workflow())
