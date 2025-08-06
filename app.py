import gradio as gr
from faster_whisper import WhisperModel
from vosk import Model, KaldiRecognizer
import json
import wave
import numpy as np
import os
import torch
import torchaudio
from chatterbox.tts import ChatterboxTTS

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = WhisperModel("turbo", device=device, compute_type="float16" if device == "cuda" else "int8")
vosk_model_path = "/app/models/vosk-model-en-us-0.22"
chatterbox_model = ChatterboxTTS.from_pretrained(device=device)

if os.path.exists(vosk_model_path):
    vosk_model = Model(vosk_model_path)
else:
    raise FileNotFoundError("VOSK model not found at /app/models/vosk-model-en-us-0.22")

def transcribe_whisper(audio_file):
    try:
        segments, info = whisper_model.transcribe(audio_file, beam_size=5, word_timestamps=True)
        
        # Convert to format similar to original whisper
        result = {
            "text": "",
            "segments": [],
            "language": info.language,
            "language_probability": info.language_probability
        }
        
        for segment in segments:
            segment_dict = {
                "id": segment.id,
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": segment.tokens,
                "temperature": segment.temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob
            }
            if hasattr(segment, 'words') and segment.words:
                segment_dict["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    }
                    for word in segment.words
                ]
            result["segments"].append(segment_dict)
            result["text"] += segment.text
        
        return result
    except Exception as e:
        return f"Whisper transcription error: {str(e)}"

def transcribe_vosk(audio_file, sample_rate=16000):
    try:
        # Initialize the recognizer with the model
        recognizer = KaldiRecognizer(vosk_model, sample_rate)
        recognizer.SetWords(True)
        
        # Open the audio file
        with open(audio_file, "rb") as audio:
            while True:
                # Read a chunk of the audio file
                data = audio.read(4000)
                if len(data) == 0:
                    break
                # Recognize the speech in the chunk
                recognizer.AcceptWaveform(data)

        result = recognizer.FinalResult()
        result_dict = json.loads(result)

        return result_dict
    except Exception as e:
        return {"error": f"VOSK transcription error: {str(e)}"}

def chatterbox_clone(text, audio_prompt=None, exaggeration=0.5, cfg_weight=0.5, temperature=1.0, random_seed=None):
    try:
        # Prepare generation parameters
        generation_params = {
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature
        }
        if random_seed is not None:
            torch.manual_seed(random_seed)  # Set seed for reproducibility if provided

        if audio_prompt:
            wav = chatterbox_model.generate(text, audio_prompt_path=audio_prompt, **generation_params)
        else:
            wav = chatterbox_model.generate(text, **generation_params)
        output_path = "output_audio.wav"
        torchaudio.save(output_path, wav, chatterbox_model.sr)
        return output_path
    except Exception as e:
        return f"Chatterbox cloning error: {str(e)}"

# Gradio Interface with custom endpoint names and new parameters
whisper_iface = gr.Interface(
    fn=transcribe_whisper,
    inputs=gr.Audio(type="filepath", label="Upload audio for Whisper transcription"),
    outputs=gr.JSON(label="Whisper Result"),
    title="OpenAI Whisper Turbo Transcription",
    api_name="whisper"
)

vosk_iface = gr.Interface(
    fn=transcribe_vosk,
    inputs=[
        gr.Audio(type="filepath", label="Upload audio for VOSK transcription"),
        gr.Number(label="Sample Rate", value=16000, precision=0)
    ],
    outputs=gr.JSON(label="VOSK Result"),
    title="VOSK Transcription",
    api_name="vosk"
)

chatterbox_iface = gr.Interface(
    fn=chatterbox_clone,
    inputs=[
        gr.Textbox(label="Text to clone"),
        gr.Audio(type="filepath", label="Reference audio (optional for voice cloning)"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="Exaggeration (emotion intensity)"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="CFG Weight (pacing control)"),
        gr.Slider(minimum=0.1, maximum=2.0, value=1.0, label="Temperature"),
        gr.Number(label="Random Seed", value=None, precision=0)
    ],
    outputs=gr.Audio(type="filepath", label="Generated Audio"),
    title="Resemble.AI Chatterbox Voice Cloning",
    api_name="chatterbox"
)

app = gr.TabbedInterface([whisper_iface, vosk_iface, chatterbox_iface], ["Whisper", "VOSK", "Chatterbox"])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)