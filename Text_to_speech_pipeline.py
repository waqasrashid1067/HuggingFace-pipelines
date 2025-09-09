from transformers import pipeline

tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")
out = tts("Hello, Hugging Face!")
with open("speech.wav", "wb") as f:
    f.write(out["audio"])
