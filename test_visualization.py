import asyncio
import numpy as np
import sys
sys.path.append('.')
from core.visualization_generator import VisualizationGenerator

async def test_visualization():
    viz_gen = VisualizationGenerator()
    
    # Create fake audio data for testing
    duration = 10  # 10 seconds
    sr = 44100
    t = np.linspace(0, duration, sr * duration)
    
    # Generate test signal (sine wave + noise)
    y = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    y = y.astype(np.float32)
    
    print(f"Testing with {len(y)/sr:.1f}s audio signal")
    
    # Test waveform generation
    waveform_overview = viz_gen._generate_waveform(y, 2000)
    print(f"Generated overview waveform: {len(waveform_overview)} points")
    
    waveform_detailed = viz_gen._generate_waveform(y, 8000)
    print(f"Generated detailed waveform: {len(waveform_detailed)} points")
    
    # Test spectrogram generation
    spectrogram = viz_gen._generate_spectrogram(y, sr)
    print(f"Generated spectrogram: {spectrogram['shape']}")
    
    print("Visualization generator test completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_visualization())
