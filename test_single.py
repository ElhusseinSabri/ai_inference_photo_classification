import requests
import time
import os

# Set API endpoint
url = "http://127.0.0.1:8000/predict"

# Folder where your test images are stored
image_folder = "test_images"  # put your .jpg or .png images in this folder

# Loop through each image
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)

        # Start timer
        start_time = time.time()

        with open(image_path, "rb") as f:
            response = requests.post(url, files={"file": f})

        # End timer
        end_time = time.time()

        latency_ms = round((end_time - start_time) * 1000, 2)

        try:
            result = response.json()
        except:
            result = "Invalid response or server error"

        print(f"{filename} -> Latency: {latency_ms} ms, Prediction: {result}")
