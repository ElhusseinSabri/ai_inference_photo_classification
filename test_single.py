import requests
import time
import os

url = "http://127.0.0.1:8000/predict"

image_folder = "test_images"

for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_folder, filename)

        start_time = time.time()

        with open(image_path, "rb") as f:
            response = requests.post(url, files={"file": f})

        end_time = time.time()

        latency_ms = round((end_time - start_time) * 1000, 2)

        try:
            result = response.json()
        except:
            result = "Invalid response or server error"

        print(f"{filename} -> Latency: {latency_ms} ms, Prediction: {result}")
