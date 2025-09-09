from transformers import pipeline

extractor = pipeline("feature-extraction", model="bert-base-uncased")
features = extractor("I love Hugging Face!")
print(len(features[0][0]))  # embedding size
