from transformers import pipeline  
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple"
)

text = "Hugging Face is based in New York and was founded by Julien on 18 oct 2022."
print(ner(text))