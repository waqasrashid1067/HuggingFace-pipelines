from transformers import pipeline

caption = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
out = caption("https://huggingface.co/datasets/mishig/sample_images/resolve/main/horse.png")
print(out)
