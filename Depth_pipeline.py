from transformers import pipeline

depth = pipeline("depth-estimation", model="Intel/dpt-large")
out = depth("https://huggingface.co/datasets/mishig/sample_images/resolve/main/parrots.png")
out["depth"].show()
