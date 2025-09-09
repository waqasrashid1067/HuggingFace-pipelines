from transformers import pipeline

segment = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")
out = segment("https://huggingface.co/datasets/mishig/sample_images/resolve/main/horse.png")
print(out[0])  # labels + masks
