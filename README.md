# Tage AI Therapist Speech Pipeline

A custom speech-to-speech pipeline for Tage's AI therapist, built to provide low-latency, high-quality voice interactions.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

3. Run the application:
```bash
python main.py
```

## Features

- Real-time audio capture from microphone
- High-quality speech-to-text using OpenAI Whisper
- Natural conversation using GPT-4
- Human-like voice synthesis using ElevenLabs
- Low-latency audio playback

## Technical Decisions

- **Audio Input/Output**: Using `sounddevice` for low-latency audio capture and playback
- **Speech-to-Text**: OpenAI Whisper for high accuracy transcription
- **LLM**: GPT-4 for natural, context-aware responses
- **Text-to-Speech**: ElevenLabs for natural, emotionally-aware voice synthesis

## Notes

This is a prototype implementation focused on quality and latency. The system is designed to feel human-like in its interactions, with particular attention paid to:
- Natural conversation flow
- Emotional intelligence in responses
- High-quality voice synthesis
- Low latency for real-time interaction