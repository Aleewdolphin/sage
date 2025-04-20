import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Audio Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
MIN_RECORD_SECONDS = 1

# Directory Configuration
RECORDINGS_DIR = "recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Voice Configuration
VOICE_MAP = {
    'therapist': 'JBFqnCBsd6RMkjVDRZzb',  # Warm, empathetic voice
    'calm': '21m00Tcm4TlvDq8ikWAM',       # Soothing, gentle voice
    'professional': 'EXAVITQu4vr4xnSDxMaL' # Clear, authoritative voice
}

# GPT Model Configuration
GPT_MODELS = {
    'gpt-4.1': 'gpt-4.1',
    'gpt-4.1-nano': 'gpt-4.1-nano',
    'gpt-4.1-mini': 'gpt-4.1-mini'
}

# TTS Configuration
TTS_MODEL = "eleven_multilingual_v2"
TTS_OUTPUT_FORMAT = "mp3_44100_128"

# System Prompt
SYSTEM_PROMPT = "You are a compassionate and empathetic AI therapist. Respond naturally and conversationally, showing understanding and care. Your response will be spoken and not read, so make sure it sounds conversational." 