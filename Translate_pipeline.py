from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
print(translator("I love machine learning", max_length=40))
