import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model_path = 'ReducedPlantDiseaseDetection.h5'
loaded_model = tf.keras.models.load_model(model_path)
print("Model loaded successfully.")

image_path = r"PlantDataset\Test\Corn__healthy\image (5).jpg"
img = image.load_img(image_path, target_size=(224, 224))
print(img)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  

predictions = loaded_model.predict(img_array)
predicted_class = np.argmax(predictions)
print("Predicted class:", predicted_class)

print("Predictions:")
print(predicted_class)
