from transformers import pipeline

classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-ks")
out = classifier("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
print(out)