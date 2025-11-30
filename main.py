from flask import Flask, jsonify
import sys

app = Flask(__name__)

print("=" * 50)
print("FLASK APP STARTING")
print("=" * 50)
print(f"Python version: {sys.version}")

try:
    print("Importing torch...")
    import torch
    print(f"✓ Torch imported: {torch.__version__}")
except Exception as e:
    print(f"✗ Torch import failed: {e}")

try:
    print("Importing torchvision...")
    from torchvision import models, transforms
    print("✓ Torchvision imported")
except Exception as e:
    print(f"✗ Torchvision import failed: {e}")

try:
    print("Loading ResNet18...")
    model = models.resnet18(pretrained=True)
    print("✓ ResNet18 loaded")
except Exception as e:
    print(f"✗ ResNet18 load failed: {e}")

print("=" * 50)
print("APP READY")
print("=" * 50)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
