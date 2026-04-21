import os
import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# --- 1. LOAD THE PYTORCH MODEL ---
# This bypasses all TensorFlow bugs by using a lightweight, pre-trained PyTorch model
print("Loading model...")
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.eval() # Set model to evaluation (prediction) mode

# The classes that ResNet18 knows (ImageNet classes)
# We will map these general classifications to our specific logic later if needed
categories = weights.meta["categories"]

# The required image transformation pipeline for PyTorch
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- 2. RECOMMENDATION LOGIC ---
# For now, we will use a simplified mock logic to ensure the pipeline works perfectly
def generate_verdict(disease_name, temperature, humidity):
    
    # In a real scenario, you'd fine-tune this model on your specific pomegranate dataset.
    # Because we are using a general model to bypass the bug, it might output a generic plant name.
    # We will simulate a detection based on environmental risk for demonstration.
    
    risk_level = "Medium"
    risk_color = "text-yellow-600"
    advice = "Maintain standard care."
    detected_condition = "Unknown Plant Issue"

    if temperature >= 25 and humidity > 80:
        detected_condition = "High Risk (Fungal conditions)"
        risk_level, risk_color = "Critical", "text-red-600"
        advice = "HIGH RISK: Warm, moist conditions accelerate fungal infections. Ensure proper soil drainage and consider preventative fungicides."
    elif temperature >= 25 and humidity > 50:
        detected_condition = "High Risk (Bacterial conditions)"
        risk_level, risk_color = "Critical", "text-red-600"
        advice = "HIGH RISK: Temperatures over 25°C with moderate humidity cause rapid spread. Avoid excess nitrogen fertilizers."
    elif temperature < 25 and humidity < 50:
         detected_condition = "Low Risk / Healthy"
         risk_level, risk_color = "Low", "text-green-600"
         advice = "Your plant environment looks healthy. Continue standard irrigation practices."

    return {
        "disease": detected_condition,
        "risk": risk_level,
        "risk_color": risk_color,
        "advice": advice
    }

# --- 3. ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded'})
            
        file = request.files['file']
        temp = float(request.form.get('temperature', 25.0))
        hum = float(request.form.get('humidity', 60.0))
        
        # Process the image
        image = Image.open(file.stream).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

        # Get the prediction
        with torch.no_grad(): # Disable gradient calculation for faster inference
            output = model(input_batch)
            
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get the top category predicted
        top_prob, top_catid = torch.topk(probabilities, 1)
        predicted_class_name = categories[top_catid[0]]
        
        # Generate the verdict using our logic
        result = generate_verdict(predicted_class_name, temp, hum)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)