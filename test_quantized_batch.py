import requests
import time
import os

url = "http://127.0.0.1:8000/predict_quantized_batch"
image_folder = "test_images"

files = []
for filename in os.listdir(image_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        with open(os.path.join(image_folder, filename), "rb") as f:
            files.append(("files", (filename, f.read(), "image/jpeg")))

start_time = time.time()
response = requests.post(url, files=files)
end_time = time.time()

print("Batch Latency:", round((end_time - start_time) * 1000, 2), "ms")
print("Response:", response.json())