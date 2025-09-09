from transformers import pipeline

txt2txt = pipeline("text2text-generation", model="google/flan-t5-small")
print(txt2txt("Translate English to French: I love Hugging Face"))
