from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import queue
import threading
from config import (
    OPENAI_API_KEY, ELEVENLABS_API_KEY, VOICE_MAP, TTS_MODEL,
    TTS_OUTPUT_FORMAT, SYSTEM_PROMPT
)

class AITherapist:
    def __init__(self, model, voice):
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.model = model
        self.voice = voice
        self.chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.text_queue = queue.Queue()
        self.tts_thread = None
        self.min_chunk_length = 1000
        self.max_chunk_length = 2000
        
    def transcribe_audio(self, audio_file):
        """Transcribe audio file to text using Whisper."""
        with open(audio_file, "rb") as audio:
            transcript = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        return transcript.text
    
    def tts_worker(self):
        """Worker thread for text-to-speech conversion and playback."""
        while True:
            text_chunk = self.text_queue.get()
            if text_chunk is None:  # Signal to stop
                break
            try:
                audio = self.elevenlabs_client.text_to_speech.convert(
                    text=text_chunk,
                    voice_id=VOICE_MAP[self.voice],
                    model_id=TTS_MODEL,
                    output_format=TTS_OUTPUT_FORMAT
                )
                play(audio)
            except Exception as e:
                print(f"Error in TTS: {e}")
    
    def start_tts_thread(self):
        """Start the TTS worker thread."""
        self.tts_thread = threading.Thread(target=self.tts_worker)
        self.tts_thread.start()
    
    def stop_tts_thread(self):
        """Stop the TTS worker thread."""
        if self.tts_thread:
            self.text_queue.put(None)
            self.tts_thread.join()
            self.tts_thread = None
    
    def find_break_point(self, text):
        """Find the last natural break point in the text."""
        for i in range(len(text)-1, -1, -1):
            if text[i] in ['.', '!', '?']:
                return i + 1
            elif text[i] in [',', ';', ':'] and len(text) > self.max_chunk_length:
                return i + 1
        return -1
    
    def process_text_chunk(self, current_text):
        """Process a chunk of text and add it to the TTS queue if appropriate."""
        if len(current_text) >= self.min_chunk_length:
            last_break = self.find_break_point(current_text)
            
            if last_break > 0:
                self.text_queue.put(current_text[:last_break])
                return current_text[last_break:]
            elif len(current_text) >= self.max_chunk_length:
                self.text_queue.put(current_text)
                return ""
        
        return current_text
    
    def generate_response(self, text):
        """Generate response from LLM and return the complete text."""
        print("Generating response...")
        
        # Add user's message to chat history
        self.chat_history.append({"role": "user", "content": text})
        
        try:
            # Stream the response from the LLM
            stream = self.openai_client.chat.completions.create(
                model=self.model,
                messages=self.chat_history,
                stream=True
            )
            
            print("AI: ", end="", flush=True)
            current_text = ""
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    current_text += content
                    current_text = self.process_text_chunk(current_text)
            
            # Send any remaining text
            if current_text.strip():
                self.text_queue.put(current_text)
            
            # Add AI's complete response to chat history
            self.chat_history.append({"role": "assistant", "content": current_text})
            return current_text
            
        except Exception as e:
            print(f"Error in LLM: {e}")
            return None
    
    def generate_and_play_response(self, text):
        """Generate response and play it in real-time."""
        try:
            self.start_tts_thread()
            response = self.generate_response(text)
            if response is None:
                print("Failed to generate response.")
        finally:
            self.stop_tts_thread()
            print()  # New line after response
    
    def clear_history(self):
        """Clear chat history while keeping the system message."""
        self.chat_history = [self.chat_history[0]] 