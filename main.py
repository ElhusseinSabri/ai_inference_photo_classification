from typing import List
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from torchvision.models.quantization import resnet18
import torch.nn.utils.prune as prune
import torch.nn as nn

print("LOADED: main.py")
app = FastAPI()

def prune_model(model, amount=0.3):
    # Apply pruning to convolutional layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)  # 30% of weights to 0
    return model
# Load once globally
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

qmodel = resnet18(pretrained=True, quantize=True)
qmodel.eval()

pruned_model = prune_model(model, amount=0.3)
pruned_quantized_model = prune_model(qmodel, amount=0.3)


with open("imagenet_classes.txt", "r") as f:
    labels = [s.strip() for s in f.readlines()]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = outputs.max(1)
    return {"class": labels[predicted.item()]}

@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        images.append(tensor)
    batch_tensor = torch.cat(images)
    with torch.no_grad():
        outputs = model(batch_tensor)
        _, predicted = torch.max(outputs, 1)
    return {"results": [{"class": labels[i.item()]} for i in predicted]}

@app.post("/predict_quantized_batch")
async def predict_quantized_batch(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        images.append(tensor)
    batch_tensor = torch.cat(images)
    with torch.no_grad():
        outputs = qmodel(batch_tensor)
        _, predicted = torch.max(outputs, 1)
    return {"results": [{"class": labels[i.item()]} for i in predicted]}

@app.post("/predict_pruned")
async def predict_pruned(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        images.append(tensor)
    batch_tensor = torch.cat(images)
    with torch.no_grad():
        outputs = pruned_model(batch_tensor)
        _, predicted = torch.max(outputs, 1)
    return {"results": [{"class": labels[i.item()]} for i in predicted]}

@app.post("/predict_pruned_quantized")
async def predict_pruned_quantized(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = transform(image).unsqueeze(0)
        images.append(tensor)
    batch_tensor = torch.cat(images)
    with torch.no_grad():
        outputs = pruned_quantized_model(batch_tensor)
        _, predicted = torch.max(outputs, 1)
    return {"results": [{"class": labels[i.item()]} for i in predicted]}