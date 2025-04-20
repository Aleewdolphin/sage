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
import argparse
from config import GPT_MODELS, VOICE_MAP
from audio import AudioRecorder, save_audio_with_timestamp, save_audio_to_temp, play_audio_file
from ai import AITherapist

# Parse command line arguments
parser = argparse.ArgumentParser(description='AI Therapist with configurable model and voice')
parser.add_argument('--model', choices=list(GPT_MODELS.keys()),
                    default='gpt-4.1-nano',
                    help='Select GPT model (default: gpt-4.1-nano)')
parser.add_argument('--voice', choices=list(VOICE_MAP.keys()),
                    default='therapist',
                    help='Select voice style: therapist (warm), calm (soothing), or professional (authoritative)')
args = parser.parse_args()

# Load environment variables from .env file
load_dotenv()

# Initialize API client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Chat history to maintain conversation context
chat_history = [
    {"role": "system", "content": "You are a compassionate and empathetic AI therapist. Respond naturally and conversationally, showing understanding and care. Your response will be spoken and not read, so make sure it sounds conversational."}
]

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
MIN_RECORD_SECONDS = 1

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI Therapist with configurable model and voice')
    parser.add_argument('--model', choices=list(GPT_MODELS.keys()),
                        default='gpt-4.1-nano',
                        help='Select GPT model (default: gpt-4.1-nano)')
    parser.add_argument('--voice', choices=list(VOICE_MAP.keys()),
                        default='therapist',
                        help='Select voice style: therapist (warm), calm (soothing), or professional (authoritative)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Initialize AI therapist
    therapist = AITherapist(GPT_MODELS[args.model], args.voice)
    recorder = AudioRecorder()
    last_recording = None
    
    print("Welcome to Tage AI Therapist.")
    print(f"Using model: {args.model}")
    print(f"Using voice: {args.voice}")
    print(f"Press Enter to start recording (minimum {MIN_RECORD_SECONDS} seconds).")
    print("Press Enter again to stop recording when you're done speaking.")
    print(f"Recordings will be saved in the '{RECORDINGS_DIR}' directory.")
    
    while True:
        print("\nOptions:")
        print("1. Start new recording")
        if last_recording:
            print("2. Replay last recording")
        print("3. Exit")
        print("4. Clear conversation history")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "3":
            break
        elif choice == "4":
            therapist.clear_history()
            print("Conversation history cleared.")
            continue
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
            text = therapist.transcribe_audio(temp_file)
            if not text.strip():
                print("No speech detected. Please try speaking louder or closer to the microphone.")
                continue
            print(f"You said: {text}")
            
            # Generate and play response
            therapist.generate_and_play_response(text)
            
        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
            print(f"\nRecording saved to: {saved_file}")

if __name__ == "__main__":
    main()