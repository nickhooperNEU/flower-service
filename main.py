from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

device = torch.device("cpu")

print("=" * 50)
print("STARTING MODEL LOADING")
print("=" * 50)

try:
    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    print("✓ ResNet18 loaded")
    
    # Replace final layer for 5 classes
    model.fc = nn.Linear(512, 5)
    print("✓ FC layer created")
    
    # Check if file exists and load weights
    if os.path.exists('classifier_head.pth'):
        print("✓ classifier_head.pth found")
        model.fc.load_state_dict(torch.load('classifier_head.pth', map_location=device))
        print("✓ Model weights loaded")
    else:
        print("⚠ classifier_head.pth NOT FOUND - using untrained weights")
    
    model = model.to(device)
    model.eval()
    print("✓ Model ready for inference")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()

print("=" * 50)
print("APP READY")
print("=" * 50)

class_labels = {0: 'Class 0', 1: 'Class 1', 2: 'Class 2', 3: 'Class 3', 4: 'Class 4'}

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_tensor = val_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return jsonify({
            'predicted_class': predicted_class,
            'class_name': class_labels[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {class_labels[i]: float(probabilities[0, i].item()) for i in range(5)}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
