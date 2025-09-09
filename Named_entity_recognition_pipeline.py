from transformers import pipeline

mask_gen = pipeline("mask-generation", model="facebook/sam-vit-base")
out = mask_gen("https://huggingface.co/datasets/mishig/sample_images/resolve/main/horse.png")
print(out[0]["mask"].size)
