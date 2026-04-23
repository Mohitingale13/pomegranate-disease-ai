import os
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
from gtts import gTTS
import io

app = Flask(__name__)

# --- 1. LOAD YOUR EXACT TRAINED MODEL ---
MODEL_PATH = 'pomegranate_disease_model.keras'

print("Rebuilding architecture and loading custom weights...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights=None)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Rescaling(1./127.5, offset=-1),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Pour your Kaggle-trained knowledge into the shell
model.load_weights(MODEL_PATH)

CLASS_NAMES = ['Alternaria', 'Anthracnose', 'Bacterial_Blight', 'Cercospora', 'Healthy']

# --- 2. LOGIC & MARATHI AUDIO GENERATION ---
def generate_audio(text):
    """Converts Marathi text to speech and encodes it for the frontend."""
    tts = gTTS(text=text, lang='mr')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return base64.b64encode(fp.read()).decode('utf-8')

def generate_verdict(disease_name, temperature, humidity):
    if disease_name == "Healthy":
        marathi_speech = "तुमचे डाळिंब निरोगी आहे. योग्य पाणी आणि खत व्यवस्थापन चालू ठेवा. पिकाला जास्त पाणी देऊ नका."
        return {
            "disease": "Healthy \u2705", "risk": "Low", "risk_color": "text-green-600",
            "advice": "Your pomegranate looks healthy. Continue standard practices.",
            "audio": generate_audio(marathi_speech)
        }
        
    advice = ""
    risk_level, risk_color = "Medium", "text-yellow-600"
    marathi_speech = ""
    
    if disease_name == "Alternaria":
        if temperature >= 20 and humidity > 85:
            risk_level, risk_color = "Critical", "text-red-600"
            advice = "HIGH RISK: Weather favorable for Alternaria spread. Improve drainage immediately and apply copper-based fungicides."
            marathi_speech = "धोका: दमट हवामानामुळे अल्टरनेरिया बुरशीचा धोका आहे. बागेत पाणी साचू देऊ नका आणि कॉपरयुक्त बुरशीनाशकाची फवारणी करा."
        else:
            advice = "Action: Improve field drainage. Remove old planting debris."
            marathi_speech = "अल्टरनेरिया आढळला आहे. बागेतील जुना कचरा नष्ट करा."
            
    elif disease_name == "Bacterial_Blight":
        if temperature >= 25 and humidity > 50:
            risk_level, risk_color = "Critical", "text-red-600"
            advice = "HIGH RISK: Temperatures cause rapid spread of Bacterial Blight. Avoid excess nitrogen. Apply Copper oxychloride."
            marathi_speech = "धोका: जास्त तापमानामुळे तेल्या रोगाचा प्रसार वेगाने होऊ शकतो. नायट्रोजनयुक्त खतांचा अतिवापर टाळा आणि कॉपर ऑक्सिक्लोराईडची फवारणी करा."
        else:
            advice = "Action: Strictly prune infected branches."
            marathi_speech = "तेल्या रोग आढळला आहे. बाधित फांद्या छाटून नष्ट करा."
            
    elif disease_name == "Anthracnose":
        if 25 <= temperature <= 30 and humidity > 80:
            risk_level, risk_color = "Critical", "text-red-600"
            advice = "HIGH RISK: Optimal breeding ground for Anthracnose. Apply fungicides like Mancozeb."
            marathi_speech = "धोका: उच्च आर्द्रतेमुळे अँथ्रॅकनोजचा धोका आहे. मॅनकोझेब सारख्या बुरशीनाशकाची फवारणी करा."
        else:
            advice = "Action: Ensure spacing between plants to reduce trapped humidity."
            marathi_speech = "अँथ्रॅकनोज आढळला आहे. झाडांमध्ये हवा खेळती राहील याची काळजी घ्या."
            
    elif disease_name == "Cercospora":
        if temperature >= 25 and humidity > 80:
            risk_level, risk_color = "Critical", "text-red-600"
            advice = "HIGH RISK: Warm, moist conditions accelerate Cercospora. Ensure proper soil drainage."
            marathi_speech = "धोका: उबदार आणि दमट वातावरणामुळे सर्कोस्पोरा रोगाचा प्रादुर्भाव वाढतो. बागेत पाणी साचू देऊ नका."
        else:
            advice = "Action: Disinfect pruning tools with bleach."
            marathi_speech = "सर्कोस्पोरा आढळला आहे. छाटणीची साधने निर्जंतुक करा."
            
    return {
        "disease": disease_name.replace('_', ' '), "risk": risk_level, 
        "risk_color": risk_color, "advice": advice, "audio": generate_audio(marathi_speech)
    }

# --- 3. ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        temp = float(request.form.get('temperature', 25.0))
        hum = float(request.form.get('humidity', 60.0))
        
        image = Image.open(file.stream).convert('RGB')
        image = image.resize((224, 224))
        img_array = np.expand_dims(np.array(image), axis=0) 
        
        predictions = model.predict(img_array)
        predicted_class_name = CLASS_NAMES[np.argmax(predictions[0])]
        
        return jsonify(generate_verdict(predicted_class_name, temp, hum))
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=True)