"""
Test script for the CNN-SVM Hybrid Acne Detection App
This script tests the API endpoints and functionality
"""

import requests
import base64
import numpy as np
from PIL import Image
import io
import os

def create_test_image():
    """Create a simple test image"""
    # Create a random RGB image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return img_base64

def test_health_endpoint():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code == 200:
            data = response.json()
            print(f"Health check passed: {data}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/model-info')
        if response.status_code == 200:
            data = response.json()
            print(f"Model info retrieved successfully:")
            print(f"  - Classes: {data['classes']}")
            print(f"  - Architecture: {data['architecture']}")
            print(f"  - Accuracy: {data['model_accuracy']}")
            print(f"  - Training samples: {data['training_samples']}")
            return True
        else:
            print(f"Model info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Model info error: {e}")
        return False

def test_prediction_endpoint():
    """Test the prediction endpoint"""
    print("\nTesting prediction endpoint...")
    try:
        # Create test image
        test_image = create_test_image()
        
        # Test with base64 image data
        payload = {'image_data': test_image}
        response = requests.post('http://localhost:5000/api/predict', json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction successful:")
            print(f"  - Predicted class: {data['prediction']}")
            print(f"  - Confidence: {data['confidence']:.2%}")
            print(f"  - All probabilities: {data['probabilities']}")
            return True
        else:
            print(f"Prediction failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Prediction error: {e}")
        return False

def test_with_real_image():
    """Test with a real image from the dataset if available"""
    print("\nTesting with real image from dataset...")
    
    # Look for any image in the dataset
    dataset_path = "acne-dataset-fin"
    test_image_path = None
    
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image_path = os.path.join(root, file)
                    break
            if test_image_path:
                break
    
    if test_image_path:
        try:
            # Load and encode the image
            with open(test_image_path, 'rb') as f:
                img_bytes = f.read()
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            # Test prediction
            payload = {'image_data': img_base64}
            response = requests.post('http://localhost:5000/api/predict', json=payload)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Real image prediction successful:")
                print(f"  - Image: {test_image_path}")
                print(f"  - Predicted class: {data['prediction']}")
                print(f"  - Confidence: {data['confidence']:.2%}")
                print(f"  - Acne info: {data['acne_info']['name']}")
                return True
            else:
                print(f"Real image prediction failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Real image test error: {e}")
            return False
    else:
        print("No real images found in dataset for testing")
        return True

def main():
    """Run all tests"""
    print("="*60)
    print("CNN-SVM Hybrid Acne Detection App - Test Suite")
    print("="*60)
    
    tests = [
        ("Health Endpoint", test_health_endpoint),
        ("Model Info Endpoint", test_model_info_endpoint),
        ("Prediction Endpoint", test_prediction_endpoint),
        ("Real Image Test", test_with_real_image)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! The app is working correctly.")
        print("You can now access the web interface at: http://localhost:5000")
    else:
        print(f"\n{total - passed} test(s) failed. Please check the issues above.")
    
    print("="*60)

if __name__ == "__main__":
    main()
