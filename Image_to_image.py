from transformers import pipeline

img2img = pipeline("image-to-image", model="hf-internal-testing/tiny-stable-diffusion-pipe")
out = img2img("https://huggingface.co/datasets/mishig/sample_images/resolve/main/horse.png", prompt="make it a cartoon")
out[0].save("cartoon.png")
