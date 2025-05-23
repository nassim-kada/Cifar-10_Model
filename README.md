# CIFAR-10 Image Classification Model

This project implements a deep learning model for image classification using the CIFAR-10 dataset. The model is built with TensorFlow/Keras and includes a user-friendly interface for testing and evaluation.

## Project Structure

```
├── Accuracy_Test.py      # Script for testing model accuracy
├── README.md            # Project documentation (this file)
├── UserInterface.py     # GUI application for model interaction
├── build.ipynb         # Jupyter notebook for model training/building
├── m_s1.h5             # Trained model weights file
└── test images/        # Directory containing test images
```

## Features

- CIFAR-10 dataset classification (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Pre-trained model with saved weights
- User interface for easy image testing
- Accuracy evaluation tools
- Interactive Jupyter notebook for training

## Installation

### Step 1: Clone or Download the Project

Download all project files to your local directory.

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv myenv

# Activate virtual environment
# On Windows:
myenv\Scripts\activate
# On macOS/Linux:
source myenv/bin/activate
```

### Step 3: Install Required Libraries

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```
## Usage

### 1. Build the model : 
Run the build file : 
open it and run every cell sequentially 
after running the build file 

You can Run the accuracy test script to evaluate the model performance:

```bash
python Accuracy_Test.py
```

### 2. Using the User Interface

Launch the GUI application for interactive image classification:

```bash
python UserInterface.py
```

The interface allows you to:
- Load images from the 'test images' directory
- Classify images using the trained model
- View prediction results with confidence scores

## Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Dataset**: CIFAR-10 (32x32 color images)
- **Classes**: 10 categories
- **Model File**: `m_s1.h5` 

### CIFAR-10 Classes:
0. Airplane
1. Automobile  
2. Bird
3. Cat
4. Deer
5. Dog
6. Frog
7. Horse
8. Ship
9. Truck

## Testing with Custom Images

1. Place your test images in the `test images/` directory
2. Run `UserInterface.py` to use the GUI
3. Load and classify your images

**Note**: For best results, images should be similar to CIFAR-10 format (small, clear objects).



[Nassim Kada]

## License

This project is created for educational purposes.

---

**Note**: Make sure all dependencies are properly installed before running the scripts. If you encounter any issues, refer to the troubleshooting section or consult the official documentation for each library.