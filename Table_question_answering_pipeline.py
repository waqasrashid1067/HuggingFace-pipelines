from transformers import pipeline
import pandas as pd

table_qa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
data = {"Country": ["USA", "Canada"], "Capital": ["Washington", "Ottawa"], "Population": ["331M", "38M"]}
table = pd.DataFrame.from_dict(data)

print(table_qa(table=table, query="What is the capital of Canada?"))
