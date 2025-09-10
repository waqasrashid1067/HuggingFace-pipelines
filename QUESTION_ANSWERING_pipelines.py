from transformers import pipeline
qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

context = "Hugging Face is a company based in New York founded in 2016."
question = "Where is Hugging Face based?"
print(qa(question=question, context=context, handle_impossible_answer=True))