from transformers import pipeline

fill = pipeline("fill-mask", model="bert-base-uncased")
print(fill("Hugging Face is a [MASK] library."))
