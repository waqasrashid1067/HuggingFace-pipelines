# from transformers import pipeline

# classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english",device=-1)
# text=["i love my country pakistan","i hate karela in my food"]
# result= classifier(text)
# print(result)
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

texts = ["I love this product!", "The service was terrible."]
results = clf(texts, batch_size=8, truncation=True, padding=True, max_length=128)
print(results)