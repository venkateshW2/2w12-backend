import subprocess
import os

def test_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        print("FFmpeg installed successfully!")
        print(f"Version: {result.stdout.split()[2]}")
        return True
    except Exception as e:
        print(f"FFmpeg installation issue: {e}")
        return False

if __name__ == "__main__":
    test_ffmpeg()
