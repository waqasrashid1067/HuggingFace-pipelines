from transformers import pipeline
QuestionAnswering = pipeline("question-answering")
QuestionAnswering(
    question= "what is my name?",
    context=("my name is waqas rashid and i am a ciomputer sceince student in islamia college peshawar "),

)