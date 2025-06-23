import asyncio
import sys
sys.path.append('.')
from core.audio_compressor import AudioCompressor

async def test_compressor():
    compressor = AudioCompressor()
    
    # Test audio info (use any audio file you have)
    # For now, just test the module loads correctly
    print("Audio compressor module loaded successfully")
    
    # Test compression settings
    settings = compressor.get_compression_settings(150_000_000, 180)
    print(f"Compression settings for 150MB, 3min file: {settings}")

if __name__ == "__main__":
    asyncio.run(test_compressor())
