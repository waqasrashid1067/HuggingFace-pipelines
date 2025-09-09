from transformers import pipeline

vid_clf = pipeline("video-classification", model="MCG-NJU/videomae-base")
out = vid_clf("https://huggingface.co/datasets/Narsil/video_demo/resolve/main/archery.mp4")
print(out)
