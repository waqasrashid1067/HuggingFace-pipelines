from transformers import pipeline

matcher = pipeline("keypoint-matching", model="huggingface/keypoint-dino")
out = matcher(
    "https://huggingface.co/datasets/mishig/sample_images/resolve/main/horse.png",
    "https://huggingface.co/datasets/mishig/sample_images/resolve/main/horse.png"
)
print(out)
