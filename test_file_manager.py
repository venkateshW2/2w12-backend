import asyncio
import sys
sys.path.append('.')
from core.file_manager import AudioFileManager

async def test_file_manager():
    fm = AudioFileManager(temp_dir="/tmp/test_audio")
    
    # Test file storage
    test_content = b"fake audio data" * 1000
    file_id = "test_123"
    
    path = await fm.store_file(file_id, test_content, "test.wav")
    print(f"Stored file at: {path}")
    
    # Test file info
    info = await fm.get_file_info(file_id)
    print(f"File info: {info}")
    
    # Test cleanup
    await fm.remove_file(file_id)
    print("File removed successfully")

if __name__ == "__main__":
    asyncio.run(test_file_manager())
