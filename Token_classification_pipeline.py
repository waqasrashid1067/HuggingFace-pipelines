from transformers import pipeline

tok_clf = pipeline("token-classification", model="dbmdz/bert-large-cased-finetuned-conll03-english")
print(tok_clf("Hugging Face is based in Paris."))
