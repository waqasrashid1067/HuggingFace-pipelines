from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("Hugging Face makes AI easy!")
print(result)