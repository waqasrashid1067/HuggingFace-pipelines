from transformers import pipeline

txt2aud = pipeline("text-to-audio", model="facebook/musicgen-small")
out = txt2aud("Generate a calm piano melody")
out[0]["audio"].save("piano.wav")
