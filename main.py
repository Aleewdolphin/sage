import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv
import tempfile
import threading
import time
from datetime import datetime
import queue

# Load environment variables from .env file
load_dotenv()

# Initialize API clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
MIN_RECORD_SECONDS = 1

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

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

def transcribe_audio(audio_file):
    with open(audio_file, "rb") as audio:
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    return transcript.text

def generate_and_play_response(text):
    """Generate response from LLM and play audio in real-time."""
    print("Generating response...")
    
    # Queue to store text chunks for TTS
    text_queue = queue.Queue()
    current_text = ""
    min_chunk_length = 1000  # Minimum characters before considering a break
    max_chunk_length = 2000  # Maximum characters before forcing a break
    
    # Start TTS thread
    def tts_worker():
        while True:
            text_chunk = text_queue.get()
            if text_chunk is None:  # Signal to stop
                break
            try:
                audio = elevenlabs_client.text_to_speech.convert(
                    text=text_chunk,
                    voice_id="JBFqnCBsd6RMkjVDRZzb",
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128"
                )
                play(audio)
            except Exception as e:
                print(f"Error in TTS: {e}")
    
    tts_thread = threading.Thread(target=tts_worker)
    tts_thread.start()
    
    try:
        # Stream the response from the LLM
        stream = openai_client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "You are a compassionate and empathetic AI therapist. Respond naturally and conversationally, showing understanding and care. Your response will be spoken and not read, so make sure it sounds conversational."},
                {"role": "user", "content": text}
            ],
            stream=True
        )
        
        print("AI: ", end="", flush=True)
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                current_text += content
                
                # Check for natural breaks, but only if we have enough text
                if len(current_text) >= min_chunk_length:
                    # Look for the last natural break point
                    last_break = -1
                    for i in range(len(current_text)-1, -1, -1):
                        if current_text[i] in ['.', '!', '?']:
                            last_break = i + 1
                            break
                        elif current_text[i] in [',', ';', ':'] and len(current_text) > max_chunk_length:
                            last_break = i + 1
                            break
                    
                    # If we found a break point, send the text up to that point
                    if last_break > 0:
                        text_queue.put(current_text[:last_break])
                        current_text = current_text[last_break:]
                    
                    # If we've accumulated too much text without a break, force a break
                    elif len(current_text) >= max_chunk_length:
                        text_queue.put(current_text)
                        current_text = ""
        
        # Send any remaining text
        if current_text.strip():
            text_queue.put(current_text)
        
    except Exception as e:
        print(f"Error in LLM: {e}")
    finally:
        # Signal TTS thread to stop
        text_queue.put(None)
        tts_thread.join()
        print()  # New line after response

def main():
    print("Welcome to Tage AI Therapist.")
    print(f"Press Enter to start recording (minimum {MIN_RECORD_SECONDS} seconds).")
    print("Press Enter again to stop recording when you're done speaking.")
    print(f"Recordings will be saved in the '{RECORDINGS_DIR}' directory.")
    recorder = AudioRecorder()
    last_recording = None
    
    while True:
        print("\nOptions:")
        print("1. Start new recording")
        if last_recording:
            print("2. Replay last recording")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "3":
            break
        elif choice == "2" and last_recording:
            print(f"\nPlaying last recording: {last_recording}")
            play_audio_file(last_recording)
            continue
        elif choice != "1":
            print("Invalid choice. Please try again.")
            continue
        
        # Start recording
        input("\nPress Enter to start recording...")
        
        # Start recording in a separate thread
        record_thread = threading.Thread(target=recorder.start_recording)
        record_thread.start()
        
        # Wait for user to stop recording
        input()
        recorder.stop_recording()
        record_thread.join()
        
        # Get the recorded audio
        audio_data = recorder.get_audio_data()
        if audio_data is None:
            print("No audio recorded. Please try again.")
            continue
        
        # Save audio to both permanent and temporary files
        saved_file = save_audio_with_timestamp(audio_data)
        temp_file = save_audio_to_temp(audio_data)
        last_recording = saved_file
        
        try:
            # Transcribe audio
            print("Transcribing...")
            text = transcribe_audio(temp_file)
            if not text.strip():
                print("No speech detected. Please try speaking louder or closer to the microphone.")
                continue
            print(f"You said: {text}")
            
            # Generate and play response
            generate_and_play_response(text)
            
        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
            print(f"\nRecording saved to: {saved_file}")

if __name__ == "__main__":
    main()