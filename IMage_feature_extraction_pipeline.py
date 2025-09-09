from transformers import pipeline

extractor = pipeline("image-feature-extraction", model="google/vit-base-patch16-224")
features = extractor("https://huggingface.co/datasets/mishig/sample_images/resolve/main/horse.png")
print(len(features[0]))  # vector size
