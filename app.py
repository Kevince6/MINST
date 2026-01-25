import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import io
from Num_Model import CNN

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(DEVICE)

try:
    model.load_state_dict(torch.load('NumRecog_method1.pth', map_location=DEVICE))
    model.eval()
    print(f"模型加载成功！运行设备: {DEVICE}")
except Exception as e:
    print(f"严重错误：模型加载失败 -> {e}")

def transform_image(image_bytes):
 
    image = Image.open(io.BytesIO(image_bytes))   
    image = image.convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return transform(image).unsqueeze(0).to(DEVICE)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': '请上传图片文件'}), 400
    
    file = request.files['file']
    
    try:
        img_bytes = file.read()
        
        tensor = transform_image(img_bytes)
        
        with torch.no_grad(): 
            outputs = model(tensor)
            
            _, predicted = torch.max(outputs, 1)
            
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence = probs[0][predicted.item()].item()

        return jsonify({
            'prediction': int(predicted.item()),
            'confidence': float(confidence),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'fail'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    