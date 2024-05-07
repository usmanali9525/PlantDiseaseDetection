from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load the trained model
model_path = 'NewReducedPlantDiseaseDetection1.h5'
loaded_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

# Class names
class_names = [
    'Corn__common_rust', 'Corn__gray_leaf_spot', 'Corn__healthy', 'Corn__northern_leaf_blight', 
    'Rice__brown_spot', 'Rice__healthy', 'Rice__leaf_blast', 'Tea__algal_leaf', 'Tea__brown_blight', 
    'Tea__healthy', 'Tea__red_leaf_spot', 'Tomato__bacterial_spot', 'Tomato__early_blight', 
    'Tomato__healthy', 'Tomato__leaf_mold', 'Tomato__septoria_leaf_spot', 'Tomato__target_spot'
]

# Function to preprocess the image
def preprocess_image(file):
    img = Image.open(file)
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    # Check number of channels
    if img_array.shape[-1] == 4:
        # Convert RGBA to RGB
        img_array = img_array[:, :, :3]
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to get predictions
def get_predictions(img_array):
    predictions = loaded_model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return class_names[predicted_class]  # Get class name from class_names list

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            img_array = preprocess_image(file)
            predicted_plant = get_predictions(img_array).split('__')[0]
            predict_disease = get_predictions(img_array).split('__')[1]
            return render_template('index.html', predicted_plant=predicted_plant, predict_disease=predict_disease)
        except Exception as e:
            return jsonify({'error': str(e)})  # Handle exception

if __name__ == '__main__':
    app.run(debug=True)
