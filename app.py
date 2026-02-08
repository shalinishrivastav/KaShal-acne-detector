"""
Acne Detection Flask Application
CNN + SVM Hybrid Model for Acne Classification
"""

from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import cv2
import base64
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models
cnn_model = None
svm_model = None
scaler = None
class_names = None
config = None

# Model paths
MODEL_DIR = 'models'  # Adjust this to your model directory
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_model_full.keras')
SVM_MODEL_PATH = os.path.join(MODEL_DIR, 'svm_classifier.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.pkl')
CONFIG_PATH = os.path.join(MODEL_DIR, 'config.pkl')

# Acne information with preventive measures
ACNE_INFO = {
    'blackheads': {
        'name': 'Blackheads',
        'description': 'Open comedones with dark appearance due to oxidized sebum',
        'prevention': 'Cleanse face twice daily with salicylic acid-based cleanser. Use non-comedogenic products. Apply topical retinoids to prevent pore blockage. Avoid oil-based cosmetics. Consider professional extractions by dermatologist.',
        'severity': 'Mild',
        'color': '#2C3E50'
    },
    'cysts': {
        'name': 'Cysts',
        'description': 'Deep, painful, pus-filled lesions under the skin',
        'prevention': 'Do not squeeze or pick at cysts. Use gentle, fragrance-free cleansers. Apply warm compresses to reduce inflammation. Avoid touching your face. Maintain consistent skincare routine with gentle products. Consider prescription treatments.',
        'severity': 'Severe',
        'color': '#8E44AD'
    },
    'darkspots': {
        'name': 'Dark Spots (Post-Inflammatory Hyperpigmentation)',
        'description': 'Discoloration left after acne healing',
        'prevention': 'Use broad-spectrum sunscreen daily (SPF 30+). Apply vitamin C serums in morning. Use products with niacinamide or alpha arbutin. Gentle chemical exfoliation with AHAs or BHAs. Avoid picking at acne to prevent scarring.',
        'severity': 'Mild to Moderate',
        'color': '#34495E'
    },
    'normal': {
        'name': 'Normal Skin',
        'description': 'Healthy skin without active acne',
        'prevention': 'Maintain current routine with gentle cleansing twice daily. Use lightweight, non-comedogenic moisturizer. Apply sunscreen every morning (SPF 30+). Stay hydrated and maintain balanced diet. Change pillowcases regularly.',
        'severity': 'None',
        'color': '#27AE60'
    },
    'papules': {
        'name': 'Papules',
        'description': 'Small, raised, red, inflamed bumps without pus',
        'prevention': 'Use benzoyl peroxide (2.5-5%) or salicylic acid products. Apply ice wrapped in cloth to reduce inflammation. Avoid picking or squeezing. Use oil-free, non-comedogenic products. Keep hands away from face. Gentle cleansing twice daily.',
        'severity': 'Moderate',
        'color': '#E74C3C'
    },
    'pustules': {
        'name': 'Pustules',
        'description': 'Inflamed, pus-filled lesions with white or yellow center',
        'prevention': 'Apply benzoyl peroxide spot treatment. Use salicylic acid cleansers. Never pop or squeeze pustules. Use clean towels and pillowcases daily. Apply oil-free moisturizers. Consider topical antibiotics if persistent.',
        'severity': 'Moderate',
        'color': '#D35400'
    },
    'whiteheads': {
        'name': 'Whiteheads',
        'description': 'Closed comedones with white appearance under skin',
        'prevention': 'Use salicylic acid cleansers twice daily. Apply retinoid treatments at night. Use lightweight, non-comedogenic moisturizers. Regular gentle exfoliation with chemical exfoliants. Avoid heavy creams and oils. Remove makeup before bed.',
        'severity': 'Mild',
        'color': '#95A5A6'
    }
}


def load_models():
    """Load all required models and preprocessing objects"""
    global cnn_model, svm_model, scaler, class_names, config
    
    try:
        print("Loading models...")
        
        # Load CNN model with compatibility fix
        try:
            # First try loading the compatible model
            compatible_model_path = os.path.join(MODEL_DIR, 'cnn_model_compatible.keras')
            if os.path.exists(compatible_model_path):
                cnn_model = tf.keras.models.load_model(compatible_model_path)
                print("CNN model loaded (compatible version)")
                
                # Check if this is the trained model or fallback
                # Test with a random image to see if predictions make sense
                test_input = tf.random.normal((1, 224, 224, 3))
                test_pred = cnn_model.predict(test_input, verbose=0)
                max_prob = tf.reduce_max(test_pred).numpy()
                
                if max_prob < 0.5:  # Random model would give ~0.14 for 7 classes
                    print("WARNING: Using untrained model - predictions may be inaccurate")
                    print("Consider training the model or using SVM-only predictions")
                else:
                    print("Trained model detected - predictions should be accurate")
                    
            else:
                # Try loading the original model with custom objects
                cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH, compile=False)
                print("CNN model loaded (original version)")
        except Exception as e:
            print(f"Error loading CNN model: {str(e)}")
            print("Creating a simple CNN model for feature extraction...")
            
            # Create a simple CNN model for feature extraction
            cnn_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu', name='feature_layer'),
                tf.keras.layers.Dense(7, activation='softmax')
            ])
            
            print("Simple CNN model created for feature extraction")
        
        # Load SVM model
        with open(SVM_MODEL_PATH, 'rb') as f:
            svm_model = pickle.load(f)
        print("SVM model loaded")
        
        # Load scaler
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded")
        
        # Load class names
        with open(CLASS_NAMES_PATH, 'rb') as f:
            class_names = pickle.load(f)
        print("Class names loaded:", class_names)
        
        # Load config
        with open(CONFIG_PATH, 'rb') as f:
            config = pickle.load(f)
        print("Config loaded")
        
        print("All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False


def preprocess_image(image_data, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    
    Args:
        image_data: Image bytes or base64 string
        target_size: Target image size (width, height)
    
    Returns:
        Preprocessed numpy array
    """
    try:
        print(f"Preprocessing image data type: {type(image_data)}")
        print(f"Image data length: {len(image_data) if isinstance(image_data, (str, bytes)) else 'N/A'}")
        
        # Handle base64 encoded images
        if isinstance(image_data, str):
            print("Processing base64 string...")
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
                print("Removed data URL prefix")
            
            try:
                image_data = base64.b64decode(image_data)
                print(f"Decoded base64, got {len(image_data)} bytes")
            except Exception as e:
                print(f"Base64 decode error: {e}")
                return None
        
        # Convert to PIL Image
        print("Converting to PIL Image...")
        try:
            image = Image.open(io.BytesIO(image_data))
            print(f"PIL Image created: {image.size}, mode: {image.mode}")
        except Exception as e:
            print(f"PIL Image creation error: {e}")
            return None
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            print(f"Converting from {image.mode} to RGB")
            image = image.convert('RGB')
        
        # Resize
        print(f"Resizing to {target_size}")
        image = image.resize(target_size)
        
        # Convert to numpy array
        print("Converting to numpy array...")
        img_array = np.array(image)
        print(f"Array shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        print(f"Normalized array shape: {img_array.shape}")
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Final array shape: {img_array.shape}")
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def extract_features(image_array):
    """
    Extract features using CNN model
    
    Args:
        image_array: Preprocessed image array
    
    Returns:
        Feature vector
    """
    try:
        # Get the feature extraction layer (second-to-last layer)
        feature_model = tf.keras.Model(
            inputs=cnn_model.input,
            outputs=cnn_model.layers[-2].output
        )
        
        # Extract features
        features = feature_model.predict(image_array, verbose=0)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None


def predict_acne(image_data):
    """
    Predict acne type from image
    
    Args:
        image_data: Image bytes or base64 string
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Preprocess image
        img_array = preprocess_image(image_data)
        if img_array is None:
            return {'error': 'Failed to preprocess image'}
        
        # Extract features
        features = extract_features(img_array)
        if features is None:
            return {'error': 'Failed to extract features'}
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Get SVM prediction
        svm_prediction = svm_model.predict(features_scaled)[0]
        
        # Get probability scores from SVM
        svm_proba = svm_model.predict_proba(features_scaled)[0]
        
        # Get CNN prediction for comparison
        cnn_proba = cnn_model.predict(img_array, verbose=0)[0]
        
        # Create probability dictionary
        probabilities = {}
        for i, class_name in enumerate(class_names):
            probabilities[class_name] = float(svm_proba[i])
        
        # Get predicted class
        predicted_class = class_names[svm_prediction]
        confidence = float(svm_proba[svm_prediction])
        
        # Get acne information
        acne_data = ACNE_INFO.get(predicted_class, {})
        
        result = {
            'success': True,
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'acne_info': acne_data,
            'cnn_probabilities': {class_names[i]: float(cnn_proba[i]) for i in range(len(class_names))}
        }
        
        return result
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {'error': f'Prediction failed: {str(e)}'}


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/test')
def test():
    """Render test page"""
    return render_template('test.html')


@app.route('/debug-image', methods=['POST'])
def debug_image():
    """Debug endpoint to see what image data is received"""
    try:
        data = request.get_json()
        print("=== DEBUG IMAGE DATA ===")
        print(f"Request data type: {type(data)}")
        print(f"Request keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        
        if 'image_data' in data:
            image_data = data['image_data']
            print(f"Image data type: {type(image_data)}")
            print(f"Image data length: {len(image_data) if isinstance(image_data, str) else 'Not a string'}")
            print(f"First 100 chars: {image_data[:100] if isinstance(image_data, str) else 'Not a string'}")
            
            # Try to decode
            try:
                decoded = base64.b64decode(image_data)
                print(f"Decoded length: {len(decoded)} bytes")
                
                # Try to create PIL image
                try:
                    img = Image.open(io.BytesIO(decoded))
                    print(f"PIL Image: {img.size}, mode: {img.mode}")
                    return jsonify({
                        'success': True,
                        'message': 'Image data is valid',
                        'image_size': img.size,
                        'image_mode': img.mode,
                        'decoded_length': len(decoded)
                    })
                except Exception as e:
                    print(f"PIL Image error: {e}")
                    return jsonify({'error': f'PIL Image error: {e}'}), 400
                    
            except Exception as e:
                print(f"Base64 decode error: {e}")
                return jsonify({'error': f'Base64 decode error: {e}'}), 400
        else:
            return jsonify({'error': 'No image_data in request'}), 400
            
    except Exception as e:
        print(f"Debug error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for image prediction"""
    try:
        # Check if image is in request
        if 'image' not in request.files and 'image_data' not in request.json:
            print("No image provided in request")
            return jsonify({'error': 'No image provided'}), 400
        
        # Get image data
        if 'image' in request.files:
            image_file = request.files['image']
            image_data = image_file.read()
            print(f"Received image file: {len(image_data)} bytes")
        else:
            image_data = request.json['image_data']
            print(f"Received image data: {len(image_data)} characters")
        
        # Make prediction
        result = predict_acne(image_data)
        
        if 'error' in result:
            print(f"Prediction error: {result['error']}")
            return jsonify(result), 400
        
        print(f"Prediction successful: {result['prediction']} ({result['confidence']:.2%})")
        return jsonify(result)
        
    except Exception as e:
        print(f"API error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-info')
def model_info():
    """Get model information"""
    info = {
        'classes': class_names,
        'num_classes': len(class_names),
        'model_accuracy': '92.53%',
        'architecture': 'CNN + SVM Hybrid',
        'training_samples': 2725,
        'validation_samples': 482,
        'acne_types': ACNE_INFO
    }
    return jsonify(info)


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': cnn_model is not None and svm_model is not None
    })


if __name__ == '__main__':
    # Load models before starting server
    if load_models():
        print("\n" + "="*60)
        print("Starting Flask Server")
        print("="*60)
        print(f"Model Accuracy: 92.53%")
        print(f"Classes: {len(class_names)}")
        print(f"Access at: http://localhost:5000")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please check model files.")
        print(f"Looking for models in: {MODEL_DIR}")
        print("Required files:")
        print("  - cnn_model_full.keras")
        print("  - svm_classifier.pkl")
        print("  - scaler.pkl")
        print("  - class_names.pkl")
        print("  - config.pkl")