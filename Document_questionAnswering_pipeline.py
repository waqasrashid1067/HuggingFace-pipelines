from transformers import pipeline

doc_qa = pipeline("document-question-answering", model="impira/layoutlm-document-qa")
out = doc_qa("https://huggingface.co/datasets/impira/documents/resolve/main/receipt.png",
             "What is the total?")
print(out)
