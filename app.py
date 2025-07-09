from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io


app = FastAPI()


model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()


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
