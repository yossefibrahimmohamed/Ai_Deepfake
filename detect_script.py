import torch
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# بناء نفس النموذج اللي دربته
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 لأن عندك فئتين: fake و real

# تحميل الأوزان فقط
model.load_state_dict(torch.load("model/fakevision_model.pt", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        if predicted.item() == 1:
            return "✅ Real Image"
        else:
            return "❌ Fake Image"
