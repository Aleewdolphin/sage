import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
from datetime import datetime
import os
import time
from config import SAMPLE_RATE, CHANNELS, MIN_RECORD_SECONDS, RECORDINGS_DIR

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = []
        self.start_time = None
        
    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_data.append(indata.copy())
    
    def start_recording(self):
        self.recording = True
        self.audio_data = []
        self.start_time = time.time()
        
        print("\nRecording started... Speak now.")
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.callback):
            while self.recording:
                elapsed = time.time() - self.start_time
                if elapsed < MIN_RECORD_SECONDS:
                    print(f"\rRecording... {elapsed:.1f}s (minimum {MIN_RECORD_SECONDS}s)", end="")
                else:
                    print(f"\rRecording... {elapsed:.1f}s (press Enter to stop)", end="")
                time.sleep(0.1)
        
        elapsed = time.time() - self.start_time
        print(f"\nRecording stopped after {elapsed:.1f} seconds")
    
    def stop_recording(self):
        self.recording = False
        return self.get_audio_data()
    
    def get_audio_data(self):
        if not self.audio_data:
            return None
        return np.concatenate(self.audio_data, axis=0)

def save_audio_with_timestamp(audio_data):
    """Save audio data to a file with timestamp in the recordings directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")
    
    sf.write(filename, audio_data, SAMPLE_RATE)
    return filename

def save_audio_to_temp(audio_data):
    """Save audio data to a temporary file for processing."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(temp_file.name, audio_data, SAMPLE_RATE)
    return temp_file.name

def play_audio_file(filename):
    """Play an audio file using sounddevice."""
    try:
        data, samplerate = sf.read(filename)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}") 