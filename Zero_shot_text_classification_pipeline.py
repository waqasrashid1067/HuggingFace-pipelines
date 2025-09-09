from transformers import pipeline

zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print(zero_shot("I love programming with Python", candidate_labels=["technology", "sports", "politics"]))
