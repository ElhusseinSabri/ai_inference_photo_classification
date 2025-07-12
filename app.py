from typing import List
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from torchvision.models.quantization import resnet18

app = FastAPI()

# Load full precision model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Load quantized model
qmodel = resnet18(pretrained=True, quantize=True)
qmodel.eval()

# Load class labels
with open("imagenet_classes.txt", "r") as f:
    labels = [s.strip() for s in f.readlines()]

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 🔹 Single image prediction
@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)
    return {"class": labels[predicted.item()]}

# 🔹 Batch prediction
# @app.post("/predict_batch")
# async def predict_batch(files: List[UploadFile] = File(...)):
#     images = []
#     for file in files:
#         image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#         tensor = transform(image).unsqueeze(0)
#         images.append(tensor)
#     batch_tensor = torch.cat(images)
#     with torch.no_grad():
#         outputs = model(batch_tensor)
#         _, predicted = torch.max(outputs, 1)
#     return {"results": [{"class": labels[idx.item()]} for idx in predicted]}
#
# # 🔹 Quantized batch prediction
# @app.post("/predict_quantized_batch")
# async def predict_quantized_batch(files: List[UploadFile] = File(...)):
#     images = []
#     for file in files:
#         image = Image.open(io.BytesIO(await file.read())).convert("RGB")
#         tensor = transform(image).unsqueeze(0)
#         images.append(tensor)
#     batch_tensor = torch.cat(images)
#     with torch.no_grad():
#         outputs = qmodel(batch_tensor)
#         _, predicted = torch.max(outputs, 1)
#     return {"results": [{"class": labels[idx.item()]} for idx in predicted]}
