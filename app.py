import gradio as gr
import whisper
from vosk import Model, KaldiRecognizer
import json
import wave
import numpy as np
import os
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

# Load models
whisper_model = whisper.load_model("turbo")
vosk_model_path = "/app/models/vosk-model-en-us-0.22"
chatterbox_model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists(vosk_model_path):
    vosk_model = Model(vosk_model_path)
else:
    raise FileNotFoundError("VOSK model not found at /app/models/vosk-model-en-us-0.22")

def transcribe_whisper(audio_file):
    try:
        result = whisper_model.transcribe(audio_file)
        return result["text"]
    except Exception as e:
        return f"Whisper transcription error: {str(e)}"

def transcribe_vosk(audio_file):
    try:
        wf = wave.open(audio_file, "rb")
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [16000, 44100]:
            return "VOSK requires mono WAV audio with 16-bit depth and 16kHz or 44.1kHz sample rate"
        recognizer = KaldiRecognizer(vosk_model, wf.getframerate())
        result = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result += json.loads(recognizer.Result())["text"] + " "
        result += json.loads(recognizer.FinalResult())["text"]
        return result.strip()
    except Exception as e:
        return f"VOSK transcription error: {str(e)}"

def chatterbox_clone(text, audio_prompt=None, exaggeration=0.5, cfg_weight=0.5):
    try:
        if audio_prompt:
            wav = chatterbox_model.generate(text, audio_prompt_path=audio_prompt, exaggeration=exaggeration, cfg_weight=cfg_weight)
        else:
            wav = chatterbox_model.generate(text, exaggeration=exaggeration, cfg_weight=cfg_weight)
        output_path = "output_audio.wav"
        torchaudio.save(output_path, wav, chatterbox_model.sr)
        return output_path
    except Exception as e:
        return f"Chatterbox cloning error: {str(e)}"

# Gradio Interface
whisper_iface = gr.Interface(
    fn=transcribe_whisper,
    inputs=gr.Audio(type="filepath", label="Upload audio for Whisper transcription"),
    outputs="text",
    title="OpenAI Whisper Turbo Transcription"
)

vosk_iface = gr.Interface(
    fn=transcribe_vosk,
    inputs=gr.Audio(type="filepath", label="Upload audio for VOSK transcription"),
    outputs="text",
    title="VOSK Transcription"
)

chatterbox_iface = gr.Interface(
    fn=chatterbox_clone,
    inputs=[
        gr.Textbox(label="Text to clone"),
        gr.Audio(type="filepath", label="Reference audio (optional for voice cloning)"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="Exaggeration (emotion intensity)"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="CFG Weight (pacing control)")
    ],
    outputs=gr.Audio(type="filepath", label="Generated Audio"),
    title="Resemble.AI Chatterbox Voice Cloning"
)

app = gr.TabbedInterface([whisper_iface, vosk_iface, chatterbox_iface], ["Whisper", "VOSK", "Chatterbox"])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)