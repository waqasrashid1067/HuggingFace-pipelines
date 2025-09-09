from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
out = asr("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac")
print(out["text"])
