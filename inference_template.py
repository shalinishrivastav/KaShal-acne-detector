
# ==================== INFERENCE SCRIPT TEMPLATE ====================
# Use this in VS Code for predictions

import numpy as np
import cv2
import pickle
from tensorflow import keras

# Load models
cnn_model = keras.models.load_model('path/to/cnn_model_full.h5')
# OR load weights:
# cnn_model.load_weights('path/to/cnn_weights.h5')

with open('path/to/svm_classifier.pkl', 'rb') as f:
    svm_classifier = pickle.load(f)

with open('path/to/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('path/to/class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

def preprocess_image(img_path, img_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2RGB)
    
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict(img_path):
    # Preprocess
    img = preprocess_image(img_path)
    
    # Extract features
    x = img
    for layer in cnn_model.layers:
        x = layer(x, training=False)
        if layer.name == 'feature_layer':
            features = x.numpy()
            break
    
    # Scale and predict
    features = scaler.transform(features)
    prediction = svm_classifier.predict(features)[0]
    probabilities = svm_classifier.predict_proba(features)[0]
    
    result = {
        'class': class_names[prediction],
        'confidence': probabilities[prediction] * 100,
        'all_probabilities': {class_names[i]: prob*100 for i, prob in enumerate(probabilities)}
    }
    
    return result

# Example usage
if __name__ == "__main__":
    result = predict('test_image.jpg')
    print(f"Predicted: {result['class']}")
    print(f"Confidence: {result['confidence']:.2f}%")
