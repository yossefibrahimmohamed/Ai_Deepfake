## Ai_Deepfake

# 🧠 FakeVision: Deepfake Image Detector

FakeVision is an AI-powered tool that detects whether an image is real or generated/manipulated by deepfake technologies. This tool leverages a fine-tuned ResNet34 convolutional neural network to classify images with reasonable confidence.

![FakeVision Banner](icon.ico) <!-- Replace with your actual banner -->

## 🚀 Features

- 🌟 Simple GUI using CustomTkinter
- 📸 Upload and analyze any `.jpg`, `.jpeg`, or `.png` image
- 📊 Outputs a clear label (`Real` or `Fake`) with confidence percentage
- 📈 Model trained with 75% accuracy on a custom dataset
- 🧩 Real-time softmax-based prediction from a trained PyTorch model

## 📷 Screenshots

This image is to test the application and how it's that

<img width="1280" alt="Screenshot 2025-05-29 065045" src="https://github.com/user-attachments/assets/20264d62-41c3-4da9-861e-d83fa1107c93" />

this images train the model ``` model/fakevision_model.pt ```

### GUI

<img width="845" alt="Screenshot 2025-05-24 025834" src="https://github.com/user-attachments/assets/7c681ef6-342c-49dc-babf-2641a815e1d2" />

<img width="847" alt="Screenshot 2025-05-24 025804" src="https://github.com/user-attachments/assets/47b2e6ad-de8d-4325-8577-ff4209aa7549" />

<img width="925" alt="Screenshot 2025-05-24 025637" src="https://github.com/user-attachments/assets/4730a089-cc2d-401c-9d72-0c0151aa71ea" />

### Model Metrics

![training_metrics](https://github.com/user-attachments/assets/9e8a7b54-4461-486c-a5e0-e2fdfb9576f6)

---

## 🔍 How It Works

1. Load the image from the file system.
2. Preprocess using torchvision transforms.
3. Feed into a fine-tuned ResNet34 model.
4. Use softmax to compute prediction probabilities.
5. Display result and confidence in the GUI.

---

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FakeVision.git
   cd FakeVision
````

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   python gui.py
   ```

---

## 🧠 Model Architecture

* Base: `ResNet34` from `torchvision.models`
* Fine-tuned last fully connected layer for binary classification
* Input image resized to `300x300`, normalized for consistency

---

## 📊 Dataset & Training

* Structure:

  ```
  dataset/
  ├── fake/
  └── real/
  ```

* Split:

  * 80% training
  * 20% validation

* Optimizer: Adam

* Loss: CrossEntropyLoss

* Epochs: 20

* Final Accuracy: **\~75%**

---

## ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Developer

**Yossef Ibrahim Mohamed**

* 💼 Programmer & Pentester
* 🎓 BSc in Computer Science
* 🏆 Projects: DeepDetect, YoutubeDownloader, Pentest\_X
* 📺 YouTube: [@yossefibrahim2001](https://youtube.com/@yossefibrahim2001)

---

## 📫 Contact

For collaborations, suggestions, or issues:

📧 [yossefibrahim@example.com](mailto:yossefibrahim@example.com)
🔗 [LinkedIn](https://www.linkedin.com/in/your-profile)

---
