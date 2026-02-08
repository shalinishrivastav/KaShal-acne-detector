"""
Quick Setup Script for Acne Detection System
Run this to verify your setup before starting the application
"""

import os
import sys

def print_banner():
    print("\n" + "="*70)
    print("🏥 KaShal- SETUP VERIFICATION")
    print("="*70 + "\n")

def check_python_version():
    print("📌 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_directory_structure():
    print("\n📂 Checking directory structure...")
    
    required_dirs = ['models', 'templates']
    optional_dirs = ['static', 'uploads']
    
    all_good = True
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}/ exists")
        else:
            print(f"   ❌ {directory}/ missing (REQUIRED)")
            all_good = False
            
    for directory in optional_dirs:
        if os.path.exists(directory):
            print(f"   ✅ {directory}/ exists")
        else:
            print(f"   ⚠️  {directory}/ missing (will be auto-created)")
            
    return all_good

def check_model_files():
    print("\n🤖 Checking model files...")
    
    required_files = [
        'models/cnn_model_full.keras',
        'models/svm_classifier.pkl',
        'models/scaler.pkl',
        'models/class_names.pkl',
        'models/config.pkl'
    ]
    
    optional_files = [
        'models/cnn_weights.weights.h5',
        'models/model_architecture.json',
        'models/training_history.pkl'
    ]
    
    all_good = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"   ✅ {os.path.basename(file_path)} ({size:.2f} MB)")
        else:
            print(f"   ❌ {os.path.basename(file_path)} (MISSING)")
            all_good = False
            
    for file_path in optional_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {os.path.basename(file_path)} ({size:.2f} MB)")
            
    return all_good

def check_template_files():
    print("\n📄 Checking template files...")
    
    if os.path.exists('templates/index.html'):
        print("   ✅ templates/index.html exists")
        return True
    else:
        print("   ❌ templates/index.html missing (REQUIRED)")
        return False

def check_dependencies():
    print("\n📦 Checking Python packages...")
    
    packages = {
        'flask': 'Flask',
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'sklearn': 'scikit-learn',
        'pickle': 'pickle (built-in)'
    }
    
    missing = []
    
    for module, name in packages.items():
        try:
            if module == 'pickle':
                import pickle
            else:
                __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (NOT INSTALLED)")
            missing.append(name)
    
    if missing:
        print(f"\n   ⚠️  Missing packages: {', '.join(missing)}")
        print("   💡 Install with: pip install -r requirements.txt")
        return False
    
    return True

def check_app_file():
    print("\n🐍 Checking application file...")
    
    if os.path.exists('app.py'):
        print("   ✅ app.py exists")
        return True
    else:
        print("   ❌ app.py missing (REQUIRED)")
        return False

def create_missing_directories():
    print("\n🔧 Creating missing directories...")
    
    dirs_to_create = ['static', 'uploads', 'static/css', 'static/js', 'static/images']
    
    for directory in dirs_to_create:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"   ✅ Created {directory}/")
            except Exception as e:
                print(f"   ❌ Failed to create {directory}/: {e}")

def print_summary(checks):
    print("\n" + "="*70)
    print("📋 SETUP SUMMARY")
    print("="*70)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {check_name:.<50} {status}")
    
    print("\n" + "="*70)
    
    if all_passed:
        print("✨ ALL CHECKS PASSED! You're ready to run the application.")
        print("\n🚀 To start the application, run:")
        print("   python app.py")
        print("\n🌐 Then open your browser and go to:")
        print("   http://localhost:5000")
    else:
        print("⚠️  SOME CHECKS FAILED. Please fix the issues above.")
        print("\n📝 Common solutions:")
        print("   1. Download model files from Google Drive to models/ folder")
        print("   2. Install dependencies: pip install -r requirements.txt")
        print("   3. Create app.py and templates/index.html files")
        print("   4. Ensure Python 3.8+ is installed")
    
    print("="*70 + "\n")

def main():
    print_banner()
    
    checks = {
        'Python Version': check_python_version(),
        'Directory Structure': check_directory_structure(),
        'Model Files': check_model_files(),
        'Template Files': check_template_files(),
        'Application File': check_app_file(),
        'Python Dependencies': check_dependencies()
    }
    
    # Create missing optional directories
    create_missing_directories()
    
    # Print summary
    print_summary(checks)
    
    return all(checks.values())

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)