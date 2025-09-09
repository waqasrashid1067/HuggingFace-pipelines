from transformers import pipeline

detector = pipeline("object-detection", model="facebook/detr-resnet-50")
out = detector("https://huggingface.co/datasets/mishig/sample_images/resolve/main/horse.png")
print(out[0])
