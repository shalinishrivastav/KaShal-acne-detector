# KaShal Acne Detector

A Flask web application that uses a hybrid CNN-SVM model to classify different types of acne from skin images.

## Features

- **Hybrid Model Architecture**: Combines CNN feature extraction with SVM classification
- **7 Acne Types**: Classifies blackheads, cysts, dark spots, normal skin, papules, pustules, and whiteheads
- **High Accuracy**: 92.53% accuracy on validation data
- **Web Interface**: User-friendly Flask web application
- **API Endpoints**: RESTful API for integration with other applications
- **Detailed Information**: Provides prevention tips and severity information for each acne type

## Model Architecture

- **CNN Feature Extractor**: Extracts 128-dimensional features from 224x224 RGB images
- **SVM Classifier**: Uses extracted features for final classification
- **Preprocessing**: Image normalization, resizing, and feature scaling
- **Training Data**: 2,725 training samples, 482 validation samples

## Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify setup** (optional):
   ```bash
   python setup.py
   ```

## Usage

### Starting the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Web Interface

1. Open your browser and go to `http://localhost:5000`
2. Upload an image of skin/acne
3. Get instant classification results with:
   - Predicted acne type
   - Confidence score
   - Prevention tips
   - Severity information

### API Usage

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Model Information
```bash
curl http://localhost:5000/api/model-info
```

#### Image Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"image_data": "base64_encoded_image_data"}'
```

### Testing

Run the comprehensive test suite:

```bash
python test_app.py
```

## Project Structure

```
cnn-svm-hybrid-model-app/
├── app.py                          # Main Flask application
├── test_app.py                     # Test suite
├── setup.py                        # Setup verification script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── models/                         # Model files
│   ├── cnn_model_compatible.keras  # Compatible CNN model
│   ├── svm_classifier.pkl          # SVM classifier
│   ├── scaler.pkl                 # Feature scaler
│   ├── class_names.pkl            # Class labels
│   └── config.pkl                 # Model configuration
├── templates/                      # HTML templates
│   └── index.html                 # Main web interface
├── static/                         # Static assets
│   ├── css/                       # Stylesheets
│   ├── js/                        # JavaScript files
│   └── images/                    # Images
├── acne-dataset-fin/              # Training dataset
│   ├── blackheads/                # Blackhead images
│   ├── cysts/                     # Cyst images
│   ├── darkspots/                # Dark spot images
│   ├── normal/                    # Normal skin images
│   ├── papules/                   # Papule images
│   ├── pustules/                  # Pustule images
│   └── whiteheads/               # Whitehead images
└── uploads/                       # Upload directory
```

## Acne Types and Information

| Type | Description | Severity | Prevention Tips |
|------|-------------|----------|-----------------|
| **Blackheads** | Open comedones with dark appearance | Mild | Salicylic acid cleanser, non-comedogenic products |
| **Cysts** | Deep, painful, pus-filled lesions | Severe | Gentle cleansers, warm compresses, avoid picking |
| **Dark Spots** | Post-inflammatory hyperpigmentation | Mild-Moderate | Sunscreen, vitamin C, gentle exfoliation |
| **Normal** | Healthy skin without active acne | None | Maintain routine, sunscreen, balanced diet |
| **Papules** | Small, raised, red, inflamed bumps | Moderate | Benzoyl peroxide, ice packs, avoid picking |
| **Pustules** | Inflamed, pus-filled lesions | Moderate | Spot treatment, clean towels, oil-free products |
| **Whiteheads** | Closed comedones with white appearance | Mild | Retinoids, chemical exfoliation, remove makeup |

## Technical Details

### Model Performance
- **Training Accuracy**: 90.66%
- **Hybrid Accuracy**: 92.53%
- **Feature Dimension**: 128
- **Input Size**: 224x224x3 RGB images
- **Classes**: 7 acne types

### Dependencies
- Flask 3.0.0
- TensorFlow 2.15.0
- NumPy 1.26.2
- Pillow 10.1.0
- OpenCV 4.8.1.78
- scikit-learn 1.3.2

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/health` | GET | Health check |
| `/api/model-info` | GET | Model information |
| `/api/predict` | POST | Image prediction |

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: 
   - Ensure all model files are present in the `models/` directory
   - Check file permissions

2. **Unicode Errors**: 
   - Fixed in the current version by removing Unicode characters from print statements

3. **Port Already in Use**: 
   - Change the port in `app.py` or kill the existing process

4. **Memory Issues**: 
   - Reduce batch size or image resolution if running on limited hardware

### Verification

Run the setup verification:
```bash
python setup.py
```

Run the test suite:
```bash
python test_app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is for educational and research purposes. Please ensure you have appropriate permissions for any medical or commercial use.

## Acknowledgments

- Dataset: Acne classification dataset
- Model Architecture: CNN-SVM hybrid approach
- Web Framework: Flask
- Deep Learning: TensorFlow/Keras
- Machine Learning: scikit-learn
