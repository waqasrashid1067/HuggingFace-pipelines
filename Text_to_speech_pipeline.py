# 
# out = tts("Hello, Hugging Face!")
# with open("speech.wav", "wb") as f:
#     f.write(out["audio"])
from transformers import pipeline
import os
import shutil
from scipy.io.wavfile import write as write_wav
import numpy as np

tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")

out = tts("Hello, my name is waqas and i am a student.")
# Define a local filename
local_filename = "intro.wav"
# Extract audio and sampling rate
audio = out["audio"]
rate = out["sampling_rate"]
# Ensure audio data is in the correct format for scipy.io.wavfile.write
# scipy.io.wavfile.write expects integer data types (e.g., int16) or float32.
# If the audio data is float32, it should be in the range [-1.0, 1.0].
# If it's an integer type, it should be within the range of that type.
# Let's assume the audio data is float32 and in the correct range.
# If it's not, further conversion might be needed.
audio = audio.astype(np.float32)

# Save the file to the local path using scipy.io.wavfile.write
# Reshape to 1D if necessary, as write_wav expects (N,) or (N, channels)
if audio.ndim > 1:
    # Assuming mono, reshape to (frames,)
    audio = audio.reshape(-1)

write_wav(local_filename, rate, audio)


# Define the destination directory in Google Drive
drive_output_dir = "/content/drive/MyDrive/audio"

# Create the directory in Google Drive if it doesn't exist
if not os.path.exists(drive_output_dir):
    os.makedirs(drive_output_dir)

# Define the full path for the destination file in Google Drive
drive_output_path = os.path.join(drive_output_dir, local_filename)

# Move the local file to Google Drive
shutil.move(local_filename, drive_output_path)

print(f"'{local_filename}' saved to {drive_output_path}")