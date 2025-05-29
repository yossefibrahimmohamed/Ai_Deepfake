import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = resnet34(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model/fakevision_model.pt", map_location=device))
model.to(device)
model.eval()

# processor using while training
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Image folder arranged
class_names = ['fake', 'real']

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence = probabilities[predicted.item()].item()  # 👈 تحويل إلى float هنا

    predicted_class = class_names[predicted.item()]
    return predicted_class, confidence
