from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english",device=-1)
text=["i love my country pakistan","i hate karela in my food"]
result= classifier(text)
print(result)
