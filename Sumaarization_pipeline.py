from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = """Hugging Face provides tools for building applications with machine learning. 
It has become the hub for open-source models, datasets, and ML developers worldwide."""
print(summarizer(text, max_length=30, min_length=10, do_sample=False))
