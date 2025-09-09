from transformers import pipeline

clf = pipeline("image-classification", model="google/vit-base-patch16-224")
print(clf("https://huggingface.co/datasets/mishig/sample_images/resolve/main/horse.png"))
