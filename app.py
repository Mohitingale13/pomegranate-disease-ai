import os
# Put TensorFlow on a memory diet BEFORE importing it
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Turn off heavy logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' # Prevent memory hoarding

import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# --- 1. LOAD THE MODEL & CLASSES ---
MODEL_PATH = 'pomegranate_disease_model.keras'

# THE BULLETPROOF FIX: Rebuild the model structure manually 
# and just load the learned weights to bypass the configuration bug.

print("Building model architecture...")
# 1. Recreate the base (without downloading weights, since we have our own)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None 
)

# 2. Recreate our custom sequence
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

print("Loading saved weights...")
# 3. Pour your trained knowledge into the empty shell
model.load_weights(MODEL_PATH)
print("Model loaded successfully!")

# These must match the exact alphabetical order from your dataset folders
CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']
# --- 2. RECOMMENDATION LOGIC ---
def generate_verdict(disease_name, temperature, humidity):
    if disease_name == "Healthy":
        return {
            "disease": "Healthy \u2705",
            "risk": "Low",
            "risk_color": "text-green-600",
            "advice": "Your pomegranate looks healthy. Continue standard irrigation and fertilizer practices. Ensure the crop is not over-watered."
        }
        
    advice = ""
    risk_level = "Medium"
    risk_color = "text-yellow-600"
    
    if disease_name == "Alternaria":
        if temperature >= 20 and humidity > 85:
            risk_level, risk_color = "Critical", "text-red-600"
            advice += "HIGH RISK: Current weather is highly favorable for Alternaria spread. "
        advice += "Action: Improve field drainage immediately. Remove old planting debris. Consider applying copper-based fungicides."
        
    elif disease_name == "Bacterial_Blight":
        if temperature >= 25 and humidity > 50:
            risk_level, risk_color = "Critical", "text-red-600"
            advice += "HIGH RISK: Temperatures over 25°C with moderate humidity cause rapid spread. "
        advice += "Action: Avoid excess nitrogen fertilizers. Apply Copper oxychloride and Streptocycline. Strictly prune infected branches."
        
    elif disease_name == "Anthracnose":
        if 25 <= temperature <= 30 and humidity > 80:
            risk_level, risk_color = "Critical", "text-red-600"
            advice += "HIGH RISK: 25-30°C with high humidity is the optimal breeding ground for Anthracnose. "
        advice += "Action: Apply fungicides like Mancozeb or Carbendazim. Ensure spacing between plants to reduce trapped humidity."
        
    elif disease_name == "Cercospora":
        if temperature >= 25 and humidity > 80:
            risk_level, risk_color = "Critical", "text-red-600"
            advice += "HIGH RISK: Warm, moist conditions accelerate Cercospora infection. "
        advice += "Action: Ensure proper soil drainage. Remove diseased fruits/twigs. Disinfect pruning tools with bleach."
        
    return {
        "disease": disease_name.replace('_', ' '),
        "risk": risk_level,
        "risk_color": risk_color,
        "advice": advice
    }

# --- 3. ROUTES ---
@app.route('/', methods=['GET'])
def index():
    # Serves the frontend HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the frontend request
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded'})
            
        file = request.files['file']
        temp = float(request.form.get('temperature', 25.0))
        hum = float(request.form.get('humidity', 60.0))
        
        # Extract features for the model
        image = Image.open(file.stream).convert('RGB') # Ensure it's in color
        image = image.resize((224, 224)) # Resize exactly as we did in Colab
        img_array = np.array(image)
        
        # The model expects a "batch" of images, so we expand dimensions
        # Shape goes from (224, 224, 3) to (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0) 
        
        # Get prediction from the ML Model
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0]) # Get the index of highest probability
        predicted_class_name = CLASS_NAMES[predicted_class_index] # Map index to name
        
        # Generate the verdict using our logic
        result = generate_verdict(predicted_class_name, temp, hum)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)